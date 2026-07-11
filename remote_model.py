"""Fireworks AI wrapper with retry + timeout safety net."""
import os, time, requests
from dataclasses import dataclass

FIREWORKS_API_URL = "https://api.fireworks.ai/inference/v1/chat/completions"

@dataclass
class RemoteResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    truncated: bool = False
    reason: str = ""
    @property
    def total_tokens(self):
        return self.prompt_tokens + self.completion_tokens

class RemoteModel:
    def __init__(self, model_name, api_key=None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("FIREWORKS_API_KEY") or None

    def generate(self, prompt, max_tokens=1024, temperature=0.0, max_retries=2, timeout=25):
        if not self.api_key:
            return RemoteResponse(text="", prompt_tokens=0, completion_tokens=0, truncated=False,
                                   reason="FIREWORKS_API_KEY not set")

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(
                    FIREWORKS_API_URL,
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json={"model": self.model_name, "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": max_tokens, "temperature": temperature},
                    timeout=timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                message = choice["message"]
                text = (message.get("content") or "").strip()
                usage = data.get("usage", {})
                return RemoteResponse(
                    text=text,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    truncated=choice.get("finish_reason") == "length",
                )
            except Exception as e:
                last_error = e
                time.sleep(1)
        return RemoteResponse(text="", prompt_tokens=0, completion_tokens=0, truncated=False,
                               reason=f"request failed after retries: {last_error}")
