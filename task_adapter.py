"""EDIT THIS on kickoff day if real task schema differs."""
import re, json as _json, ast, operator
from dataclasses import dataclass

@dataclass
class VerdictResult:
    accept: bool
    reason: str

_ANSWER_INSTRUCTION = (
    "Answer directly and precisely. If the question expects a specific format "
    "(a number, a label, JSON, a short phrase), respond with ONLY that -- no "
    "extra explanation, preamble, or restatement of the question.\n\n"
)

def _raw_task_text(task):
    return (task.get("prompt") or task.get("question") or task.get("input", "")).strip()

def build_prompt(task):
    return _ANSWER_INSTRUCTION + _raw_task_text(task)

def strip_code_fences(text):
    stripped = text.strip()
    match = re.match(r"^```(?:json)?\\s*\\n?(.*?)\\n?```$", stripped, re.DOTALL)
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
    cleaned = prompt_text.replace("\u00d7", "*").replace("x", "*")
    match = re.search(r"(-?\\d+\\.?\\d*)\\s*([*+\\-/])\\s*(-?\\d+\\.?\\d*)", cleaned)
    if not match:
        return None
    expr = f"{match.group(1)} {match.group(2)} {match.group(3)}"
    try:
        result = _safe_eval(ast.parse(expr, mode="eval"))
        return int(result) if result == int(result) else result
    except Exception:
        return None

def verify_numeric_answer(task, answer_text):
    numbers = re.findall(r"-?\\d+\\.?\\d*", answer_text)
    if len(numbers) == 1:
        return VerdictResult(True, "single clean numeric answer found")
    if len(numbers) == 0:
        return VerdictResult(False, "no numeric answer found")
    return VerdictResult(False, "ambiguous: multiple numbers in answer")

def verify_numeric_consistency(answer_a, answer_b):
    nums_a = re.findall(r"-?\\d+\\.?\\d*", answer_a)
    nums_b = re.findall(r"-?\\d+\\.?\\d*", answer_b)
    if len(nums_a) == 1 and len(nums_b) == 1:
        try:
            if abs(float(nums_a[0]) - float(nums_b[0])) < 1e-6:
                return VerdictResult(True, f"numeric self-consistent ({nums_a[0]})")
        except ValueError:
            pass
    return VerdictResult(False, f"numeric answers disagree ({nums_a} vs {nums_b})")

def verify_structured_answer(task, answer_text, required_keys=None):
    cleaned = strip_code_fences(answer_text)
    try:
        parsed = _json.loads(cleaned)
    except _json.JSONDecodeError:
        return VerdictResult(False, "not valid JSON")
    if required_keys and not all(k in parsed for k in required_keys):
        return VerdictResult(False, f"missing required keys: {required_keys}")
    return VerdictResult(True, "valid structured output")

def verify_choice_answer(task, answer_text, valid_choices):
    normalized = answer_text.strip().strip(".").lower()
    matches = [c for c in valid_choices if c.lower() == normalized or c.lower() in normalized]
    if len(matches) == 1:
        return VerdictResult(True, f"matched choice: {matches[0]}")
    return VerdictResult(False, f"no unambiguous match among {valid_choices}")

def verify_self_consistency(answer_a, answer_b, similarity_threshold=0.8):
    from difflib import SequenceMatcher
    ratio = SequenceMatcher(None, answer_a.strip().lower(), answer_b.strip().lower()).ratio()
    if ratio >= similarity_threshold:
        return VerdictResult(True, f"self-consistent (ratio={ratio:.2f})")
    return VerdictResult(False, f"low self-consistency (ratio={ratio:.2f})")

_UNCERTAIN_PHRASES = (
    "i don't know", "i'm not sure", "unclear", "n/a",
    "i cannot", "i can't", "as an ai", "i don't have enough information",
    "insufficient information", "not enough information",
)

def generic_verifier(task, answer_text):
    stripped = answer_text.strip()
    if len(stripped) < 2:
        return VerdictResult(False, "answer too short / empty")
    lowered = stripped.lower()
    if lowered in _UNCERTAIN_PHRASES or any(lowered.startswith(p) for p in _UNCERTAIN_PHRASES):
        return VerdictResult(False, "model expressed uncertainty or refused to answer")
    if lowered == _raw_task_text(task).lower():
        return VerdictResult(False, "answer just echoes the question back")
    return VerdictResult(True, "passed generic sanity check")

def verify(task, answer_text, second_sample=None):
    task_type = task.get("type", "").lower()
    if task_type in ("math", "numeric", "arithmetic"):
        result = verify_numeric_answer(task, answer_text)
        if result.accept and second_sample is not None:
            return verify_numeric_consistency(answer_text, second_sample)
        return result
    elif task_type in ("json", "structured", "extraction"):
        result = verify_structured_answer(task, answer_text, task.get("required_keys"))
    elif task_type in ("classification", "multiple_choice", "mcq"):
        result = verify_choice_answer(task, answer_text, task.get("choices", []))
    else:
        result = generic_verifier(task, answer_text)

    if result.accept and second_sample is not None:
        consistency = verify_self_consistency(answer_text, second_sample)
        if not consistency.accept:
            return consistency
    return result
