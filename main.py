"""Entrypoint: /input/tasks.json -> /output/results.json"""
import json, os
from local_model import LocalModel
from remote_model import RemoteModel
from router import RoutingAgent
from token_meter import TokenMeter

INPUT_PATH = os.environ.get("TASKS_INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("RESULTS_OUTPUT_PATH", "/output/results.json")
LOCAL_MODEL_NAME = os.environ.get("LOCAL_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
REMOTE_MODEL_NAME = os.environ.get("REMOTE_MODEL_NAME", "accounts/fireworks/models/gpt-oss-120b")

def main():
    with open(INPUT_PATH) as f:
        tasks = json.load(f)

    local = LocalModel(model_name=LOCAL_MODEL_NAME)
    remote = RemoteModel(model_name=REMOTE_MODEL_NAME)
    meter = TokenMeter()
    agent = RoutingAgent(local, remote, meter, use_self_consistency=False)

    results = agent.solve_batch(tasks)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {len(results)} results to {OUTPUT_PATH}")
    print(meter.summary())

if __name__ == "__main__":
    main()
