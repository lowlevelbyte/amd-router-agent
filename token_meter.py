"""Tracks token usage: local (free) vs remote (billable)."""
from dataclasses import dataclass, field

@dataclass
class TokenMeter:
    local_tokens: int = 0
    remote_tokens: int = 0
    remote_calls: int = 0
    local_calls: int = 0
    log: list = field(default_factory=list)

    def record_local(self, task_id, tokens):
        self.local_tokens += tokens
        self.local_calls += 1
        self.log.append({"task_id": task_id, "route": "local", "tokens": tokens})

    def record_remote(self, task_id, tokens):
        self.remote_tokens += tokens
        self.remote_calls += 1
        self.log.append({"task_id": task_id, "route": "remote", "tokens": tokens})

    @property
    def billable_tokens(self):
        return self.remote_tokens

    def summary(self):
        total_calls = self.local_calls + self.remote_calls
        return {
            "billable_tokens": self.billable_tokens,
            "local_tokens_free": self.local_tokens,
            "remote_calls": self.remote_calls,
            "local_calls": self.local_calls,
            "pct_solved_locally": round(100 * self.local_calls / total_calls, 1) if total_calls else 0.0,
        }
