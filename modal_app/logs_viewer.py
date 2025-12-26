"""View and export interaction logs from Modal volume."""
from __future__ import annotations

import modal

app = modal.App("hi-moe-logs-viewer")

logs_volume = modal.Volume.from_name("hi-moe-logs", create_if_missing=True)
LOGS_PATH = "/logs"

viewer_image = modal.Image.debian_slim(python_version="3.11")


@app.function(
    image=viewer_image,
    volumes={LOGS_PATH: logs_volume},
)
def list_logs() -> list[dict]:
    """List all log files in the volume."""
    import os

    logs = []
    if not os.path.exists(LOGS_PATH):
        return logs

    for filename in sorted(os.listdir(LOGS_PATH)):
        if filename.endswith(".jsonl"):
            path = f"{LOGS_PATH}/{filename}"
            size = os.path.getsize(path)
            # Count lines
            with open(path) as f:
                line_count = sum(1 for _ in f)
            logs.append({
                "filename": filename,
                "size_bytes": size,
                "interactions": line_count,
            })

    return logs


@app.function(
    image=viewer_image,
    volumes={LOGS_PATH: logs_volume},
)
def read_logs(filename: str, limit: int = 10) -> list[dict]:
    """Read interactions from a specific log file."""
    import json

    path = f"{LOGS_PATH}/{filename}"
    interactions = []

    with open(path) as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            interactions.append(json.loads(line))

    return interactions


@app.function(
    image=viewer_image,
    volumes={LOGS_PATH: logs_volume},
)
def export_for_training(output_filename: str = "training_data.jsonl") -> dict:
    """Export logs in training format (problem/reasoning/solution)."""
    import json
    import os

    output_path = f"{LOGS_PATH}/{output_filename}"
    exported = 0

    with open(output_path, "w") as out:
        for filename in os.listdir(LOGS_PATH):
            if not filename.startswith("interactions_"):
                continue

            with open(f"{LOGS_PATH}/{filename}") as f:
                for line in f:
                    entry = json.loads(line)

                    # Extract user message and response
                    user_msg = ""
                    for msg in entry.get("messages", []):
                        if msg.get("role") == "user":
                            user_msg = msg.get("content", "")
                            break

                    response = entry.get("response", "")

                    # Skip incomplete responses
                    if entry.get("finish_reason") != "stop":
                        continue

                    # Format for training
                    training_entry = {
                        "domain": "general",  # Can be classified later
                        "problem": user_msg,
                        "reasoning": response,
                        "solution": "",  # Extract code if present
                    }
                    out.write(json.dumps(training_entry) + "\n")
                    exported += 1

    logs_volume.commit()
    return {"exported": exported, "output": output_path}


@app.local_entrypoint()
def main(action: str = "list", filename: str = "", limit: int = 10):
    """View interaction logs.

    Actions:
        list: List all log files
        read: Read interactions from a file
        export: Export logs for training

    Examples:
        modal run modal_app/logs_viewer.py --action=list
        modal run modal_app/logs_viewer.py --action=read --filename=interactions_2024-01-15.jsonl
        modal run modal_app/logs_viewer.py --action=export
    """
    import json

    if action == "list":
        logs = list_logs.remote()
        if not logs:
            print("No log files found.")
        else:
            print(f"Found {len(logs)} log file(s):")
            total_interactions = 0
            for log in logs:
                print(f"  - {log['filename']}: {log['interactions']} interactions ({log['size_bytes']} bytes)")
                total_interactions += log["interactions"]
            print(f"\nTotal interactions: {total_interactions}")

    elif action == "read":
        if not filename:
            print("Error: --filename required")
            return

        interactions = read_logs.remote(filename, limit)
        print(f"Showing {len(interactions)} interactions from {filename}:\n")
        for i, entry in enumerate(interactions):
            print(f"--- Interaction {i+1} ({entry['request_id']}) ---")
            print(f"Timestamp: {entry['timestamp']}")
            print(f"Adapter: {entry.get('adapter') or 'base'}")
            print(f"User: {entry['messages'][-1]['content'][:100]}...")
            print(f"Response: {entry['response'][:200]}...")
            print(f"Tokens: {entry['usage']['total_tokens']}")
            print()

    elif action == "export":
        result = export_for_training.remote()
        print(f"Exported {result['exported']} interactions to {result['output']}")

    else:
        print(f"Unknown action: {action}")
        print("Valid actions: list, read, export")
