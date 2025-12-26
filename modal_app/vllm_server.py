"""vLLM server deployment on Modal with LoRA support."""
from __future__ import annotations

import modal
from pydantic import BaseModel

app = modal.App("hi-moe-inference")


# Pydantic models for API (defined at module level for FastAPI)
class ChatRequest(BaseModel):
    model: str = "base"  # "base" or adapter name
    messages: list[dict]
    max_tokens: int = 2048
    temperature: float = 0.7


class ChatResponse(BaseModel):
    id: str
    choices: list[dict]
    usage: dict

# Persistent volume for LoRA adapters only
adapter_volume = modal.Volume.from_name("hi-moe-adapters", create_if_missing=True)

MODEL_ID = "Qwen/QwQ-32B-AWQ"
TOKENIZER_ID = "Qwen/QwQ-32B"  # Use base model tokenizer (has vocab.json)
ADAPTERS_PATH = "/adapters"
MODEL_DIR = "/models"


def download_models():
    """Download model weights during image build to eliminate cold start download."""
    from huggingface_hub import snapshot_download
    import os

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Download AWQ model
    print(f"Downloading {MODEL_ID}...")
    snapshot_download(
        MODEL_ID,
        local_dir=f"{MODEL_DIR}/{MODEL_ID}",
        ignore_patterns=["*.md", "*.txt"],
    )

    # Download tokenizer from base model
    print(f"Downloading tokenizer from {TOKENIZER_ID}...")
    snapshot_download(
        TOKENIZER_ID,
        local_dir=f"{MODEL_DIR}/{TOKENIZER_ID}",
        allow_patterns=["tokenizer*", "vocab*", "merges*", "*.json"],
    )
    print("Model download complete!")


# Container image with vLLM - bake model into image for fast cold starts
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.6.0",
        "huggingface_hub",
        "hf_transfer",
        "fastapi",
        "pydantic",
        "transformers",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(download_models)  # Bake model into image
)


@app.cls(
    gpu="A100-80GB",  # 32B model with LoRA needs 80GB
    image=vllm_image,
    volumes={
        ADAPTERS_PATH: adapter_volume,
    },
    timeout=3600,  # 1 hour max request time
    scaledown_window=300,  # Keep warm for 5 min
)
@modal.concurrent(max_inputs=100)
class VLLMServer:
    @modal.enter()
    def start_server(self):
        """Initialize vLLM engine on container start."""
        from vllm import LLM
        from vllm.lora.request import LoRARequest
        import os

        # Use local paths from baked-in model
        model_path = f"{MODEL_DIR}/{MODEL_ID}"
        tokenizer_path = f"{MODEL_DIR}/{TOKENIZER_ID}"

        print(f"Loading model from {model_path}...")
        print(f"Loading tokenizer from {tokenizer_path}...")

        # Discover available adapters
        self.adapters = {}
        if os.path.exists(ADAPTERS_PATH):
            for adapter_name in os.listdir(ADAPTERS_PATH):
                adapter_path = f"{ADAPTERS_PATH}/{adapter_name}"
                if os.path.isdir(adapter_path):
                    self.adapters[adapter_name] = adapter_path

        # Initialize vLLM with LoRA support
        # Use local paths for fast cold starts (model baked into image)
        self.llm = LLM(
            model=model_path,
            tokenizer=tokenizer_path,
            quantization="awq",
            enable_lora=True,
            max_loras=8,
            max_lora_rank=64,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            trust_remote_code=True,
            enforce_eager=True,  # Disable torch.compile to save memory
        )

        self.lora_id_counter = 1
        self.lora_requests = {}

        # Pre-register discovered adapters
        for name, path in self.adapters.items():
            self.lora_requests[name] = LoRARequest(
                lora_name=name,
                lora_int_id=self.lora_id_counter,
                lora_local_path=path,
            )
            self.lora_id_counter += 1
            print(f"Registered adapter: {name}")

    def _generate_internal(
        self,
        prompt: str,
        adapter_name: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: list[str] | None = None,
    ) -> dict:
        """Internal generate method callable from ASGI endpoints."""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
        )

        lora_request = None
        if adapter_name and adapter_name in self.lora_requests:
            lora_request = self.lora_requests[adapter_name]

        outputs = self.llm.generate(
            [prompt],
            sampling_params,
            lora_request=lora_request,
        )

        output = outputs[0]
        return {
            "text": output.outputs[0].text,
            "prompt_tokens": len(output.prompt_token_ids),
            "completion_tokens": len(output.outputs[0].token_ids),
            "adapter_used": adapter_name,
            "finish_reason": output.outputs[0].finish_reason,
        }

    def _chat_internal(
        self,
        messages: list[dict],
        adapter_name: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> dict:
        """Internal chat method callable from ASGI endpoints."""
        # Format messages as ChatML
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        return self._generate_internal(
            prompt=prompt,
            adapter_name=adapter_name,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|im_end|>"],
        )

    @modal.method()
    def generate(
        self,
        prompt: str,
        adapter_name: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: list[str] | None = None,
    ) -> dict:
        """Generate completion with optional LoRA adapter."""
        return self._generate_internal(prompt, adapter_name, max_tokens, temperature, top_p, stop)

    @modal.method()
    def chat(
        self,
        messages: list[dict],
        adapter_name: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> dict:
        """Chat completion with ChatML formatting."""
        return self._chat_internal(messages, adapter_name, max_tokens, temperature)

    @modal.method()
    def list_adapters(self) -> list[str]:
        """List available LoRA adapters."""
        return list(self.lora_requests.keys())

    @modal.method()
    def health(self) -> dict:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "model": MODEL_ID,
            "adapters": list(self.lora_requests.keys()),
            "max_model_len": 4096,
        }

    @modal.asgi_app()
    def serve(self):
        """Serve OpenAI-compatible API endpoint on the same container."""
        from fastapi import FastAPI

        api = FastAPI(title="hi-moe vLLM API")

        @api.post("/v1/chat/completions")
        def chat_completions(body: ChatRequest) -> ChatResponse:
            adapter = None if body.model == "base" else body.model

            result = self._chat_internal(
                messages=body.messages,
                adapter_name=adapter,
                max_tokens=body.max_tokens,
                temperature=body.temperature,
            )

            return ChatResponse(
                id="chatcmpl-" + format(abs(hash(result["text"])), "x")[:8],
                choices=[{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result["text"],
                    },
                    "finish_reason": result["finish_reason"],
                }],
                usage={
                    "prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": result["completion_tokens"],
                    "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
                },
            )

        @api.get("/v1/models")
        def list_models():
            models = [{"id": "base", "object": "model"}]
            for adapter in self.lora_requests.keys():
                models.append({"id": adapter, "object": "model"})
            return {"data": models}

        @api.get("/health")
        def health():
            return {
                "status": "healthy",
                "model": MODEL_ID,
                "adapters": list(self.lora_requests.keys()),
                "max_model_len": 4096,
            }

        return api
