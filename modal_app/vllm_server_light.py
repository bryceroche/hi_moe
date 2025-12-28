"""Lightweight vLLM server for routing/planning tiers (hi_moe-4gd).

Uses Qwen3-4B-Instruct on T4 GPU for fast, cheap inference on non-code tasks.
This handles Architect planning and Dispatcher routing - no LoRA adapters needed.
"""
from __future__ import annotations

import modal
from pydantic import BaseModel

app = modal.App("hi-moe-router")


class ChatRequest(BaseModel):
    model: str = "base"
    messages: list[dict]
    max_tokens: int = 512  # Routing/planning needs less output
    temperature: float = 0.3  # Lower temp for deterministic routing


class ChatResponse(BaseModel):
    id: str
    choices: list[dict]
    usage: dict


MODEL_ID = "Qwen/Qwen3-4B-Instruct"
MODEL_DIR = "/models"


def download_model():
    """Download model during image build."""
    from huggingface_hub import snapshot_download
    import os

    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Downloading {MODEL_ID}...")
    snapshot_download(
        MODEL_ID,
        local_dir=f"{MODEL_DIR}/{MODEL_ID}",
        ignore_patterns=["*.md", "*.txt"],
    )
    print("Model download complete!")


# Lightweight image - no LoRA dependencies needed
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
    .run_function(download_model)
)


@app.cls(
    gpu="T4",  # Cheap GPU - 4B model fits easily
    image=vllm_image,
    timeout=600,  # 10 min max
    scaledown_window=300,  # 5 min warm (cheaper than main server)
)
@modal.concurrent(max_inputs=50)
class LightVLLMServer:
    @modal.enter()
    def start_server(self):
        """Initialize vLLM engine."""
        from vllm import LLM

        model_path = f"{MODEL_DIR}/{MODEL_ID}"
        print(f"Loading model from {model_path}...")

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            trust_remote_code=True,
            enforce_eager=True,
        )
        print("Router model loaded!")

    def _generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.95,
        stop: list[str] | None = None,
    ) -> dict:
        """Generate completion."""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
        )

        outputs = self.llm.generate([prompt], sampling_params)
        output = outputs[0]

        return {
            "text": output.outputs[0].text,
            "prompt_tokens": len(output.prompt_token_ids),
            "completion_tokens": len(output.outputs[0].token_ids),
            "finish_reason": output.outputs[0].finish_reason,
        }

    def _chat(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> dict:
        """Chat completion with ChatML formatting."""
        import uuid

        # Format as ChatML with /no_think for fast response
        prompt = ""
        for i, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"]
            # Add /no_think to last user message
            if role == "user" and i == len(messages) - 1:
                content = content + " /no_think"
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        result = self._generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|im_end|>"],
        )

        return {
            "id": f"router-{uuid.uuid4().hex[:8]}",
            **result,
        }

    @modal.method()
    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> dict:
        """Chat completion endpoint."""
        return self._chat(messages, max_tokens, temperature)

    @modal.method()
    def health(self) -> dict:
        """Health check."""
        return {
            "status": "healthy",
            "model": MODEL_ID,
            "type": "router",
        }

    @modal.asgi_app()
    def serve(self):
        """HTTP API endpoint."""
        from fastapi import FastAPI

        api = FastAPI(title="hi-moe Router API")

        @api.post("/v1/chat/completions")
        def chat_completions(body: ChatRequest) -> ChatResponse:
            result = self._chat(
                messages=body.messages,
                max_tokens=body.max_tokens,
                temperature=body.temperature,
            )

            return ChatResponse(
                id=result["id"],
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
            return {"data": [{"id": "base", "object": "model"}]}

        @api.get("/health")
        def health():
            return {
                "status": "healthy",
                "model": MODEL_ID,
                "type": "router",
            }

        return api
