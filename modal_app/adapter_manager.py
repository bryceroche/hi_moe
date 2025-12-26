"""LoRA adapter management for Modal - download from HuggingFace or list available.

IMPORTANT: AWQ Compatibility
---------------------------
The vLLM server uses QwQ-32B-AWQ (AWQ quantized). LoRA adapters trained on
the full-precision QwQ-32B-Preview model are NOT compatible with the AWQ model
due to different weight representations.

To use LoRA adapters, you must either:
1. Train adapters specifically on the AWQ model (using training.py)
2. Use the full-precision base model (requires more GPU memory)

vLLM also requires adapters to only target supported modules:
- Supported: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- NOT supported: embed_tokens, lm_head, modules_to_save
"""
from __future__ import annotations

import modal

app = modal.App("hi-moe-adapters")

# Volume for LoRA adapters (shared with inference server)
adapter_volume = modal.Volume.from_name("hi-moe-adapters", create_if_missing=True)

ADAPTERS_PATH = "/adapters"

# Available QwQ-32B adapters from HuggingFace
AVAILABLE_ADAPTERS = {
    "abliterated": "pipihand01/QwQ-32B-Preview-abliterated-lora-rank32",
    "instruct": "FUfu99/QwQ-32B-Preview-Instruct_lora",
    "creed": "phxdev/qwq-32b-lora-creed",
    "ftr-phase1": "FBSUZUKI/Qwen_QwQ-32B-Preview_ftr_lora_phase1",
}

adapter_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(
    image=adapter_image,
    volumes={ADAPTERS_PATH: adapter_volume},
    timeout=1800,  # 30 min max
)
def download_adapter(repo_id: str, adapter_name: str | None = None) -> dict:
    """Download a LoRA adapter from HuggingFace to the Modal volume.

    Args:
        repo_id: HuggingFace repo (e.g., "pipihand01/QwQ-32B-Preview-abliterated-lora-rank32")
        adapter_name: Local name for the adapter (defaults to repo name)
    """
    from huggingface_hub import snapshot_download
    import os

    # Derive adapter name from repo if not provided
    if adapter_name is None:
        adapter_name = repo_id.split("/")[-1]

    output_path = f"{ADAPTERS_PATH}/{adapter_name}"

    print(f"Downloading {repo_id} to {output_path}...")

    snapshot_download(
        repo_id,
        local_dir=output_path,
        ignore_patterns=["*.md", "*.txt", ".gitattributes"],
    )

    # List downloaded files
    files = os.listdir(output_path)
    print(f"Downloaded files: {files}")

    # Commit the volume changes
    adapter_volume.commit()

    return {
        "adapter_name": adapter_name,
        "repo_id": repo_id,
        "path": output_path,
        "files": files,
    }


@app.function(
    image=adapter_image,
    volumes={ADAPTERS_PATH: adapter_volume},
)
def list_adapters() -> list[dict]:
    """List all adapters in the Modal volume."""
    import os

    adapters = []

    if not os.path.exists(ADAPTERS_PATH):
        return adapters

    for name in os.listdir(ADAPTERS_PATH):
        path = f"{ADAPTERS_PATH}/{name}"
        if os.path.isdir(path):
            files = os.listdir(path)
            # Check for required LoRA files
            has_config = "adapter_config.json" in files
            has_weights = any(f.endswith(".safetensors") or f.endswith(".bin") for f in files)

            adapters.append({
                "name": name,
                "path": path,
                "files": files,
                "valid": has_config and has_weights,
            })

    return adapters


@app.function(
    image=adapter_image,
    volumes={ADAPTERS_PATH: adapter_volume},
)
def delete_adapter(adapter_name: str) -> dict:
    """Delete an adapter from the Modal volume."""
    import shutil
    import os

    path = f"{ADAPTERS_PATH}/{adapter_name}"

    if not os.path.exists(path):
        return {"success": False, "error": f"Adapter {adapter_name} not found"}

    shutil.rmtree(path)
    adapter_volume.commit()

    return {"success": True, "deleted": adapter_name}


@app.local_entrypoint()
def main(
    action: str = "list",
    adapter: str = "",
    name: str = "",
):
    """Manage LoRA adapters.

    Actions:
        list: List all adapters in the volume
        download: Download an adapter from HuggingFace
        delete: Delete an adapter
        available: Show available pre-defined adapters

    Examples:
        modal run modal_app/adapter_manager.py --action=list
        modal run modal_app/adapter_manager.py --action=available
        modal run modal_app/adapter_manager.py --action=download --adapter=abliterated
        modal run modal_app/adapter_manager.py --action=download --adapter=pipihand01/QwQ-32B-Preview-abliterated-lora-rank32 --name=my-adapter
        modal run modal_app/adapter_manager.py --action=delete --adapter=my-adapter
    """
    if action == "list":
        adapters = list_adapters.remote()
        if not adapters:
            print("No adapters found in volume.")
        else:
            print(f"Found {len(adapters)} adapter(s):")
            for a in adapters:
                status = "valid" if a["valid"] else "INVALID"
                print(f"  - {a['name']} [{status}]: {len(a['files'])} files")

    elif action == "available":
        print("Available pre-defined adapters:")
        for short_name, repo_id in AVAILABLE_ADAPTERS.items():
            print(f"  - {short_name}: {repo_id}")
        print("\nDownload with: modal run modal_app/adapter_manager.py --action=download --adapter=<name>")

    elif action == "download":
        if not adapter:
            print("Error: --adapter required (use short name or full HuggingFace repo)")
            return

        # Check if it's a short name
        repo_id = AVAILABLE_ADAPTERS.get(adapter, adapter)
        adapter_name = name if name else None

        print(f"Downloading {repo_id}...")
        result = download_adapter.remote(repo_id, adapter_name)
        print(f"Downloaded: {result['adapter_name']}")
        print(f"  Path: {result['path']}")
        print(f"  Files: {result['files']}")

    elif action == "delete":
        if not adapter:
            print("Error: --adapter required")
            return

        result = delete_adapter.remote(adapter)
        if result["success"]:
            print(f"Deleted adapter: {result['deleted']}")
        else:
            print(f"Error: {result['error']}")

    else:
        print(f"Unknown action: {action}")
        print("Valid actions: list, available, download, delete")
