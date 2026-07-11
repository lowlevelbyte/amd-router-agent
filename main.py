"""Entrypoint: /input/tasks.json -> /output/results.json (Fireworks-only, no local model)."""
import json, os
from remote_model import RemoteModel, resolve_allowed_models
from task_adapter import build_prompt, verify, try_deterministic_math, strip_code_fences
from token_meter import TokenMeter

INPUT_PATH = os.environ.get("INPUT_PATH") or os.environ.get("TASKS_INPUT_PATH") or "/input/tasks.json"
OUTPUT_PATH = os.environ.get("OUTPUT_PATH") or os.environ.get("RESULTS_OUTPUT_PATH") or "/output/results.json"

def _write_results(results):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

def solve_task(task, remote, meter):
    task_id = task.get("id", "unknown")
    prompt = build_prompt(task)
    task_type = task.get("type", "").lower()

    if task_type in ("math", "numeric", "arithmetic"):
        computed = try_deterministic_math(prompt)
        if computed is not None:
            return {"task_id": task_id, "answer": str(computed), "route": "computed",
                    "reason": "solved via zero-cost deterministic calculator"}

    resp = remote.generate(prompt, reasoning_effort="none")
    meter.record_remote(task_id, resp.total_tokens)
    answer_text = strip_code_fences(resp.text) if task_type == "json" else resp.text
    verdict = verify(task, answer_text)
    reason = verdict.reason

    if not verdict.accept:
        # reasoning_effort="none" plus a tight max_tokens can come back
        # truncated/empty on harder tasks -- retry once with reasoning
        # enabled and more headroom before accepting a failed answer.
        retry_resp = remote.generate(prompt, max_tokens=2048, reasoning_effort=None)
        meter.record_remote(task_id, retry_resp.total_tokens)
        retry_text = strip_code_fences(retry_resp.text) if task_type == "json" else retry_resp.text
        retry_verdict = verify(task, retry_text)
        answer_text, reason = retry_text, f"retry with reasoning enabled: {retry_verdict.reason}"

    return {"task_id": task_id, "answer": answer_text.strip(), "route": "remote", "reason": reason}

def main():
    allowed_models = resolve_allowed_models()
    print(f"[startup] FIREWORKS_BASE_URL={os.environ.get('FIREWORKS_BASE_URL') or '(not set, using default)'}")
    print(f"[startup] ALLOWED_MODELS={os.environ.get('ALLOWED_MODELS') or '(not set)'}")
    print(f"[startup] Resolved model fallback chain={allowed_models}")

    results = []
    try:
        with open(INPUT_PATH) as f:
            tasks = json.load(f)

        remote = RemoteModel(allowed_models=allowed_models)
        meter = TokenMeter()

        for task in tasks:
            try:
                results.append(solve_task(task, remote, meter))
            except Exception as e:
                # Absolute last resort -- guarantees every task_id gets SOME
                # entry in results.json, even if everything else failed.
                results.append({
                    "task_id": task.get("id", "unknown"),
                    "answer": "",
                    "route": "failed",
                    "reason": f"unrecoverable error: {e}",
                })

        _write_results(results)
        print(f"Wrote {len(results)} results to {OUTPUT_PATH}")
        print(meter.summary())
    except Exception as e:
        # Top-level safety net -- guarantees results.json exists even if
        # tasks.json couldn't be read or the remote model failed to construct.
        _write_results(results)
        print(f"Wrote {len(results)} results to {OUTPUT_PATH} after top-level failure: {e}")

if __name__ == "__main__":
    main()
