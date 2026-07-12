"""Entrypoint: /input/tasks.json -> /output/results.json (Fireworks-only, no local model)."""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, wait
from remote_model import RemoteModel, resolve_allowed_models
from task_adapter import build_prompt, verify, try_deterministic_math, strip_code_fences
from token_meter import TokenMeter

INPUT_PATH = os.environ.get("INPUT_PATH") or os.environ.get("TASKS_INPUT_PATH") or "/input/tasks.json"
OUTPUT_PATH = os.environ.get("OUTPUT_PATH") or os.environ.get("RESULTS_OUTPUT_PATH") or "/output/results.json"
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "20"))
MAX_TOTAL_SECONDS = float(os.environ.get("MAX_TOTAL_SECONDS", "90"))

def _write_results(results):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

def _flush_and_exit(code=0):
    # ThreadPoolExecutor registers a process-wide atexit hook that joins
    # EVERY worker thread ever created, regardless of shutdown(wait=False)
    # on any individual executor -- so a still-running HTTP call can block
    # normal interpreter exit even after we've abandoned it below. os._exit
    # skips that atexit machinery entirely, which is the only way to make
    # MAX_TOTAL_SECONDS an actual hard ceiling on wall-clock time.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)

def solve_task(task, remote, meter):
    task_id = task.get("id", "unknown")
    prompt = build_prompt(task)
    task_type = task.get("type", "").lower()

    if task_type in ("math", "numeric", "arithmetic"):
        computed = try_deterministic_math(prompt)
        if computed is not None:
            return {"task_id": task_id, "answer": str(computed), "route": "computed",
                    "reason": "solved via zero-cost deterministic calculator"}

    # Fastest possible single call: guaranteed completion matters more than
    # accuracy right now -- a timed-out submission scores zero regardless.
    resp = remote.generate(prompt, max_tokens=512, reasoning_effort="none")
    meter.record_remote(task_id, resp.total_tokens)
    answer_text = strip_code_fences(resp.text) if task_type == "json" else resp.text
    verdict = verify(task, answer_text)

    return {"task_id": task_id, "answer": answer_text.strip(), "route": "remote", "reason": verdict.reason}

def _solve_with_safety_net(task, remote, meter):
    start = time.time()
    try:
        result = solve_task(task, remote, meter)
    except Exception as e:
        # Absolute last resort -- guarantees every task_id gets SOME
        # entry in results.json, even if everything else failed.
        result = {
            "task_id": task.get("id", "unknown"),
            "answer": "",
            "route": "failed",
            "reason": f"unrecoverable error: {e}",
        }
    return result, time.time() - start

def main():
    allowed_models = resolve_allowed_models()
    print(f"[startup] FIREWORKS_BASE_URL={os.environ.get('FIREWORKS_BASE_URL') or '(not set, using default)'}")
    print(f"[startup] ALLOWED_MODELS={os.environ.get('ALLOWED_MODELS') or '(not set)'}")
    print(f"[startup] Resolved model fallback chain={allowed_models}")
    print(f"[startup] MAX_WORKERS={MAX_WORKERS}")
    print(f"[startup] MAX_TOTAL_SECONDS={MAX_TOTAL_SECONDS}")

    results = []
    try:
        with open(INPUT_PATH) as f:
            tasks = json.load(f)
        print(f"[startup] Loaded {len(tasks)} tasks")

        remote = RemoteModel(allowed_models=allowed_models)
        meter = TokenMeter()

        results = [None] * len(tasks)
        start_time = time.time()

        executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        futures = {
            executor.submit(_solve_with_safety_net, task, remote, meter): i
            for i, task in enumerate(tasks)
        }

        remaining = max(0, MAX_TOTAL_SECONDS - (time.time() - start_time))
        done, not_done = wait(futures.keys(), timeout=remaining)

        completed_count = 0
        for future in done:
            index = futures[future]
            result, elapsed = future.result()
            results[index] = result
            completed_count += 1
            print(f"Completed {completed_count}/{len(tasks)} tasks (last took {elapsed:.1f}s)")

        timed_out_count = 0
        for future in not_done:
            index = futures[future]
            timed_out_task = tasks[index]
            results[index] = {
                "task_id": timed_out_task.get("id", "unknown"),
                "answer": "",
                "route": "timed_out",
                "reason": "did not complete within MAX_TOTAL_SECONDS budget",
            }
            timed_out_count += 1

        # Abandon anything still running -- don't block process exit on it.
        executor.shutdown(wait=False, cancel_futures=True)

        _write_results(results)
        total_elapsed = time.time() - start_time
        print(f"Finished in {total_elapsed:.1f}s: {completed_count}/{len(tasks)} completed, {timed_out_count}/{len(tasks)} timed out")
        print(meter.summary())
        _flush_and_exit(0)
    except Exception as e:
        # Top-level safety net -- guarantees results.json exists even if
        # tasks.json couldn't be read or the remote model failed to construct.
        _write_results(results)
        print(f"Wrote {len(results)} results to {OUTPUT_PATH} after top-level failure: {e}")
        _flush_and_exit(0)

if __name__ == "__main__":
    main()
