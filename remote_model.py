"""Fireworks AI wrapper with retry + timeout safety net."""
import json, os, time, requests
from dataclasses import dataclass

DEFAULT_FIREWORKS_API_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
DEFAULT_REMOTE_MODEL_NAME = "accounts/fireworks/models/gpt-oss-120b"

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

def _resolve_base_url():
    base_url = os.environ.get("FIREWORKS_BASE_URL")
    if not base_url:
        return DEFAULT_FIREWORKS_API_URL
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    # NOTE: we don't yet know the exact shape of the harness-injected
    # FIREWORKS_BASE_URL -- this heuristic may need adjusting once we can
    # see the actual value (check the [startup] log lines in main.py).
    if base_url.endswith("/inference/v1") or base_url.endswith("/v1"):
        return base_url + "/chat/completions"
    return base_url + "/inference/v1/chat/completions"

def _parse_allowed_models(raw):
    if not raw:
        return []
    raw = raw.strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(m).strip() for m in parsed if str(m).strip()]
        if isinstance(parsed, str) and parsed.strip():
            return [parsed.strip()]
    except (json.JSONDecodeError, TypeError):
        pass
    if "," in raw:
        return [m.strip() for m in raw.split(",") if m.strip()]
    return [raw]

def resolve_remote_model_name():
    allowed = _parse_allowed_models(os.environ.get("ALLOWED_MODELS"))
    default = allowed[0] if allowed else DEFAULT_REMOTE_MODEL_NAME
    return os.environ.get("REMOTE_MODEL_NAME", default)

class RemoteModel:
    def __init__(self, model_name, api_key=None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("FIREWORKS_API_KEY") or None
        self.base_url = _resolve_base_url()

    def generate(self, prompt, max_tokens=1024, temperature=0.0, max_retries=2, timeout=25):
        if not self.api_key:
            return RemoteResponse(text="", prompt_tokens=0, completion_tokens=0, truncated=False,
                                   reason="FIREWORKS_API_KEY not set")

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(
                    self.base_url,
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
