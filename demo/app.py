"""
TokenCascade -- Live Demo
AMD Developer Hackathon: ACT II, Track 1

Single-file Streamlit app so it deploys with zero import-path issues.
Runs the real cascade logic (calculator -> local model -> verify -> remote)
live, using Fireworks AI for the remote tier. The local tier attempts a
small model and degrades gracefully with a clear message if the hosting
environment can't support it (Streamlit Cloud free tier has ~1GB RAM, no GPU).
"""
import ast
import json as _json
import operator
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import requests
import streamlit as st

st.set_page_config(page_title="TokenCascade", page_icon="\U0001F9E9", layout="wide")

# ---------------------------------------------------------------------------
# Token meter
# ---------------------------------------------------------------------------

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

    def summary(self):
        total_calls = self.local_calls + self.remote_calls
        return {
            "billable_tokens": self.remote_tokens,
            "local_tokens_free": self.local_tokens,
            "remote_calls": self.remote_calls,
            "local_calls": self.local_calls,
            "pct_solved_locally": round(100 * self.local_calls / total_calls, 1) if total_calls else 0.0,
        }


# ---------------------------------------------------------------------------
# Task adapter: prompt building + verifiers (same logic as the submission)
# ---------------------------------------------------------------------------

@dataclass
class VerdictResult:
    accept: bool
    reason: str


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```$", stripped, re.DOTALL)
    return match.group(1).strip() if match else stripped


_SAFE_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
             ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    raise ValueError("unsafe expression")


def try_deterministic_math(prompt_text):
    cleaned = prompt_text.replace("\u00d7", "*")
    match = re.search(r"(-?\d+\.?\d*)\s*([*+\-/])\s*(-?\d+\.?\d*)", cleaned)
    if not match:
        return None
    expr = f"{match.group(1)} {match.group(2)} {match.group(3)}"
    try:
        result = _safe_eval(ast.parse(expr, mode="eval"))
        return int(result) if result == int(result) else result
    except Exception:
        return None


def verify_numeric_answer(answer_text):
    numbers = re.findall(r"-?\d+\.?\d*", answer_text)
    if len(numbers) == 1:
        return VerdictResult(True, "single clean numeric answer found")
    if len(numbers) == 0:
        return VerdictResult(False, "no numeric answer found")
    return VerdictResult(False, "ambiguous: multiple numbers in answer")


def verify_numeric_consistency(answer_a, answer_b):
    nums_a = re.findall(r"-?\d+\.?\d*", answer_a)
    nums_b = re.findall(r"-?\d+\.?\d*", answer_b)
    if len(nums_a) == 1 and len(nums_b) == 1:
        try:
            if abs(float(nums_a[0]) - float(nums_b[0])) < 1e-6:
                return VerdictResult(True, f"numeric self-consistent ({nums_a[0]})")
        except ValueError:
            pass
    return VerdictResult(False, f"numeric answers disagree ({nums_a} vs {nums_b})")


def verify_structured_answer(answer_text, required_keys=None):
    cleaned = strip_code_fences(answer_text)
    try:
        parsed = _json.loads(cleaned)
    except _json.JSONDecodeError:
        return VerdictResult(False, "not valid JSON")
    if required_keys and not all(k in parsed for k in required_keys):
        return VerdictResult(False, f"missing required keys: {required_keys}")
    return VerdictResult(True, "valid structured output")


def verify_choice_answer(answer_text, valid_choices):
    normalized = answer_text.strip().strip(".").lower()
    matches = [c for c in valid_choices if c.lower() == normalized or c.lower() in normalized]
    if len(matches) == 1:
        return VerdictResult(True, f"matched choice: {matches[0]}")
    return VerdictResult(False, f"no unambiguous match among {valid_choices}")


def verify_self_consistency(answer_a, answer_b, threshold=0.8):
    ratio = SequenceMatcher(None, answer_a.strip().lower(), answer_b.strip().lower()).ratio()
    if ratio >= threshold:
        return VerdictResult(True, f"self-consistent (ratio={ratio:.2f})")
    return VerdictResult(False, f"low self-consistency (ratio={ratio:.2f})")


def generic_verifier(answer_text):
    stripped = answer_text.strip()
    if len(stripped) < 2:
        return VerdictResult(False, "answer too short / empty")
    if stripped.lower() in {"i don't know", "i'm not sure", "unclear", "n/a"}:
        return VerdictResult(False, "model expressed uncertainty")
    return VerdictResult(True, "passed generic sanity check")


def verify(task_type, answer_text, choices=None, required_keys=None, second_sample=None):
    if task_type in ("math", "numeric", "arithmetic"):
        result = verify_numeric_answer(answer_text)
        if result.accept and second_sample is not None:
            return verify_numeric_consistency(answer_text, second_sample)
        return result
    elif task_type in ("json", "structured", "extraction"):
        result = verify_structured_answer(answer_text, required_keys)
    elif task_type in ("classification", "multiple_choice", "mcq"):
        result = verify_choice_answer(answer_text, choices or [])
    else:
        result = generic_verifier(answer_text)

    if result.accept and second_sample is not None:
        consistency = verify_self_consistency(answer_text, second_sample)
        if not consistency.accept:
            return consistency
    return result


# ---------------------------------------------------------------------------
# Local model (small, CPU-friendly attempt with graceful degradation)
# ---------------------------------------------------------------------------

LOCAL_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@st.cache_resource(show_spinner="Loading local model (first run only, ~1-2 min)...")
def load_local_model():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_NAME, dtype=torch.float32, device_map="cpu")
        return tokenizer, model
    except Exception as e:
        return None, str(e)


def local_generate(prompt, max_tokens=150, temperature=0.0):
    tokenizer, model = load_local_model()
    if tokenizer is None:
        return None, 0, 0

    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    prompt_tokens = inputs["input_ids"].shape[1]

    gen_kwargs = dict(max_new_tokens=max_tokens, pad_token_id=tokenizer.eos_token_id)
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature)
    else:
        gen_kwargs.update(do_sample=False)

    out_ids = model.generate(**inputs, **gen_kwargs)
    gen_ids = out_ids[0][prompt_tokens:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text, prompt_tokens, len(gen_ids)


# ---------------------------------------------------------------------------
# Remote model (Fireworks AI)
# ---------------------------------------------------------------------------

FIREWORKS_API_URL = "https://api.fireworks.ai/inference/v1/chat/completions"


def remote_generate(prompt, api_key, model_name, max_tokens=512, timeout=30):
    try:
        resp = requests.post(
            FIREWORKS_API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model_name, "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": max_tokens, "temperature": 0.0},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        text = (data["choices"][0]["message"].get("content") or "").strip()
        usage = data.get("usage", {})
        return text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), None
    except Exception as e:
        return "", 0, 0, str(e)


# ---------------------------------------------------------------------------
# Router (the cascade)
# ---------------------------------------------------------------------------

def solve(task_type, prompt, api_key, remote_model_name, choices=None, required_keys=None, meter=None):
    steps = []

    if task_type in ("math", "numeric", "arithmetic"):
        computed = try_deterministic_math(prompt)
        if computed is not None:
            return {"answer": str(computed), "route": "computed",
                    "reason": "solved via zero-cost deterministic calculator", "steps": steps}

    local_text, p_tok, c_tok = local_generate(prompt)
    local_unavailable = local_text is None

    if local_unavailable:
        steps.append("Local tier unavailable in this hosted environment (likely memory-constrained) -- escalating.")
        verdict = VerdictResult(False, "local model unavailable in demo host")
    else:
        if meter:
            meter.record_local("demo", p_tok + c_tok)
        steps.append(f"Local model answered: {local_text[:200]!r}")

        second_sample = None
        need_consistency = task_type in ("math", "numeric", "arithmetic") or task_type not in (
            "json", "structured", "extraction", "classification", "multiple_choice", "mcq")
        if need_consistency:
            second_text, p2, c2 = local_generate(prompt, temperature=0.7)
            if second_text is not None:
                if meter:
                    meter.record_local("demo", p2 + c2)
                second_sample = second_text
                steps.append(f"Second local sample (for consistency check): {second_text[:200]!r}")

        verdict = verify(task_type, local_text, choices=choices, required_keys=required_keys,
                          second_sample=second_sample)
        steps.append(f"Verifier result: accept={verdict.accept}, reason={verdict.reason}")

    if verdict.accept and not local_unavailable:
        final_answer = strip_code_fences(local_text) if task_type == "json" else local_text
        return {"answer": final_answer.strip(), "route": "local", "reason": verdict.reason, "steps": steps}

    remote_text, p_tok, c_tok, error = remote_generate(prompt, api_key, remote_model_name)
    if error:
        steps.append(f"Remote call failed: {error}")
        return {"answer": "", "route": "failed", "reason": f"escalated but remote call failed: {error}", "steps": steps}

    if meter:
        meter.record_remote("demo", p_tok + c_tok)
    final_answer = strip_code_fences(remote_text) if task_type == "json" else remote_text
    return {"answer": final_answer.strip(), "route": "remote",
            "reason": f"escalated: {verdict.reason}", "steps": steps}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

if "meter" not in st.session_state:
    st.session_state.meter = TokenMeter()

st.title("\U0001F9E9 TokenCascade")
st.caption("Live demo -- AMD Developer Hackathon: ACT II, Track 1: Hybrid Token-Efficient Routing Agent")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Fireworks API Key", type="password",
                             help="Only used for this session, never stored or logged.")
    remote_model_name = st.text_input("Remote model", value="accounts/fireworks/models/gpt-oss-120b")

    st.divider()
    st.subheader("How it works")
    st.markdown(
        "1. **Deterministic calculator** -- zero-cost, for clean math expressions\n"
        "2. **Local model** (small model here, Qwen2.5-1.5B in the real submission) attempts the task\n"
        "3. **Verifier** checks the answer (format check / self-consistency / exact match)\n"
        "4. **Remote escalation** (Fireworks gpt-oss-120b) only if the verifier rejects the local answer"
    )
    st.divider()
    st.subheader("Session token usage")
    st.json(st.session_state.meter.summary())
    if st.button("Reset session stats"):
        st.session_state.meter = TokenMeter()
        st.rerun()

st.markdown("### Try it")

presets = {
    "Math (clean expression)": ("math", "What is 347 * 289? Answer with only the number.", None, None),
    "Math (word problem)": ("math", "A shop had 84 apples. They sold 27, got 45 more, then sold 18. How many are left? Answer with only the number.", None, None),
    "Sentiment classification": ("classification", "Classify the sentiment as positive or negative: 'The product broke after two days and support never replied.'", "positive,negative", None),
    "JSON extraction": ("json", "Extract the name and age from this text as JSON with keys 'name' and 'age': 'John is 34 years old.'", None, "name,age"),
    "Open-ended question": ("generic", "Explain in one sentence why the sky is blue.", None, None),
}

cols = st.columns(len(presets))
selected_preset = None
for col, (label, _) in zip(cols, presets.items()):
    if col.button(label, use_container_width=True):
        selected_preset = label

if "task_type" not in st.session_state:
    st.session_state.task_type = "math"
    st.session_state.prompt_text = ""
    st.session_state.choices_text = ""
    st.session_state.keys_text = ""

if selected_preset:
    t, p, c, k = presets[selected_preset]
    st.session_state.task_type = t
    st.session_state.prompt_text = p
    st.session_state.choices_text = c or ""
    st.session_state.keys_text = k or ""

task_type = st.selectbox(
    "Task type", ["math", "classification", "json", "generic"],
    index=["math", "classification", "json", "generic"].index(st.session_state.task_type),
)
prompt = st.text_area("Task prompt", value=st.session_state.prompt_text, height=100)

choices = required_keys = None
if task_type == "classification":
    choices_text = st.text_input("Valid choices (comma-separated)", value=st.session_state.choices_text)
    choices = [c.strip() for c in choices_text.split(",") if c.strip()]
elif task_type == "json":
    keys_text = st.text_input("Required JSON keys (comma-separated)", value=st.session_state.keys_text)
    required_keys = [k.strip() for k in keys_text.split(",") if k.strip()]

run = st.button("Run through TokenCascade", type="primary")

if run:
    if not api_key:
        st.error("Please enter a Fireworks API key in the sidebar first.")
    elif not prompt.strip():
        st.error("Please enter a task prompt.")
    else:
        with st.spinner("Running cascade..."):
            result = solve(task_type, prompt, api_key, remote_model_name,
                            choices=choices, required_keys=required_keys,
                            meter=st.session_state.meter)

        route_colors = {"computed": "green", "local": "blue", "remote": "orange", "failed": "red"}
        route = result["route"]
        st.markdown(f"**Route taken:** :{route_colors.get(route, 'gray')}[{route.upper()}]")
        st.markdown(f"**Reason:** {result['reason']}")
        st.markdown("**Answer:**")
        st.code(result["answer"] or "(empty)", language=None)

        with st.expander("Show step-by-step trace"):
            for step in result["steps"]:
                st.write("-", step)

        st.divider()
        st.markdown("**Session token usage so far:**")
        st.json(st.session_state.meter.summary())

st.divider()
st.caption(
    "Note: this hosted demo uses a small local model (Qwen2.5-0.5B) due to free-tier hosting constraints "
    "(no GPU, limited RAM). The actual hackathon submission runs Qwen2.5-1.5B-Instruct on a dedicated AMD GPU. "
    "If the local tier is unavailable in this environment, tasks escalate to the remote model automatically -- "
    "the routing logic itself is identical to the submitted agent."
)