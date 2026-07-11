"""Cascade controller: computed -> local -> verify -> remote."""
from local_model import LocalModel
from remote_model import RemoteModel
from token_meter import TokenMeter
from task_adapter import build_prompt, verify, try_deterministic_math, strip_code_fences

class RoutingAgent:
    def __init__(self, local_model, remote_model, meter, use_self_consistency=False):
        self.local_model = local_model
        self.remote_model = remote_model
        self.meter = meter
        self.use_self_consistency = use_self_consistency

    def solve(self, task):
        task_id = task.get("id", "unknown")
        prompt = build_prompt(task)
        task_type = task.get("type", "").lower()

        if task_type in ("math", "numeric", "arithmetic"):
            computed = try_deterministic_math(prompt)
            if computed is not None:
                return {"task_id": task_id, "answer": str(computed), "route": "computed",
                        "reason": "solved via zero-cost deterministic calculator"}

        local_resp = self.local_model.generate(prompt)
        self.meter.record_local(task_id, local_resp.total_tokens)

        need_consistency = self.use_self_consistency or task_type in ("math", "numeric", "arithmetic")
        second_sample_text = None
        if need_consistency:
            _, resp_b = self.local_model.generate_twice_for_agreement(prompt)
            second_sample_text = resp_b.text
            self.meter.record_local(task_id, resp_b.total_tokens)

        verdict = verify(task, local_resp.text, second_sample=second_sample_text)

        if verdict.accept:
            final_answer = strip_code_fences(local_resp.text) if task_type == "json" else local_resp.text
            return {"task_id": task_id, "answer": final_answer.strip(), "route": "local", "reason": verdict.reason}

        remote_resp = self.remote_model.generate(prompt)
        self.meter.record_remote(task_id, remote_resp.total_tokens)
        final_answer = strip_code_fences(remote_resp.text) if task_type == "json" else remote_resp.text
        return {"task_id": task_id, "answer": final_answer.strip(), "route": "remote", "reason": f"escalated: {verdict.reason}"}

    def solve_batch(self, tasks):
        return [self.solve(task) for task in tasks]
