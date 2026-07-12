"""Fireworks AI wrapper with retry + timeout safety net + model fallback chain."""
import os, time, requests
from dataclasses import dataclass

DEFAULT_FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_REMOTE_MODEL_NAME = "accounts/fireworks/models/gpt-oss-120b"

@dataclass
class RemoteResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    truncated: bool = False
    reason: str = ""
    model_used: str = ""
    @property
    def total_tokens(self):
        return self.prompt_tokens + self.completion_tokens

def _resolve_base_url():
    base_url = (os.environ.get("FIREWORKS_BASE_URL") or DEFAULT_FIREWORKS_BASE_URL).rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return base_url + "/chat/completions"

def _parse_allowed_models(raw):
    if not raw:
        return []
    raw = raw.strip()
    if not raw:
        return []
    if "," in raw:
        return [m.strip() for m in raw.split(",") if m.strip()]
    return [raw]

def resolve_allowed_models():
    """Ordered list of model IDs to try, first-to-last (the fallback chain)."""
    allowed = _parse_allowed_models(os.environ.get("ALLOWED_MODELS"))
    override = os.environ.get("REMOTE_MODEL_NAME")
    if override:
        return [override] + [m for m in allowed if m != override]
    return allowed or [DEFAULT_REMOTE_MODEL_NAME]

class RemoteModel:
    def __init__(self, allowed_models=None, api_key=None):
        self.allowed_models = allowed_models or resolve_allowed_models()
        self.api_key = api_key or os.environ.get("FIREWORKS_API_KEY") or None
        self.base_url = _resolve_base_url()

    def generate(self, prompt, max_tokens=1024, temperature=0.0, max_retries=0, timeout=20, reasoning_effort="none"):
        if not self.api_key:
            return RemoteResponse(text="", prompt_tokens=0, completion_tokens=0, truncated=False,
                                   reason="FIREWORKS_API_KEY not set")

        last_error = None
        for model_name in self.allowed_models:
            for attempt in range(max_retries + 1):
                try:
                    payload = {
                        "model": model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    }
                    if reasoning_effort is not None:
                        payload["reasoning_effort"] = reasoning_effort

                    resp = requests.post(
                        self.base_url,
                        headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                        json=payload,
                        timeout=timeout,
                    )
                    if resp.status_code == 404:
                        # Model not deployed under this account -- no point
                        # retrying the same model, fall through to the next
                        # one in the ALLOWED_MODELS chain.
                        last_error = f"model '{model_name}' returned 404 (not deployed)"
                        break

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
                        model_used=model_name,
                    )
                except Exception as e:
                    last_error = e
                    time.sleep(1)

        return RemoteResponse(text="", prompt_tokens=0, completion_tokens=0, truncated=False,
                               reason=f"all models in ALLOWED_MODELS failed: {last_error}")
