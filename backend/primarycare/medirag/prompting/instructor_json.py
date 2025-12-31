from __future__ import annotations

import json
import re
from typing import Any, Dict, List


# ============================================================
# 2) Robust JSON extraction: fixes "Extra data" / trailing text
# ============================================================

def extract_first_json_object(text: str) -> str:  # noqa: C901
    """
    Extracts the first complete JSON object {...} from a string using brace balancing.
    Safe with braces inside quoted strings.
    """
    start = text.find("{")
    if start == -1:
        return text.strip()

    in_string = False
    escape = False
    depth = 0

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1].strip()

    end = text.rfind("}")
    if end > start:
        return text[start : end + 1].strip()
    return text[start:].strip()


# ============================================================
# 3) Instruction JSON schema + normalization + validation
#    (NO clarifying_questions anywhere)
# ============================================================

REQUIRED_KEYS = {
    "verification_level",
    "answer_mode",
    "required_sections",
    "constraints",
    "context_plan",
}


def normalize_instruction_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only REQUIRED_KEYS, fill missing with safe defaults.
    This prevents extra/unwanted keys (like clarifying_questions) from propagating.
    """
    out: Dict[str, Any] = {k: obj.get(k) for k in REQUIRED_KEYS}

    if out.get("verification_level") is None:
        out["verification_level"] = 1
    if out.get("answer_mode") is None:
        out["answer_mode"] = "abstractive"
    if out.get("required_sections") is None:
        out["required_sections"] = ["Direct answer", "Evidence summary", "Limitations"]
    if out.get("constraints") is None:
        out["constraints"] = ["Avoid absolute claims", "State uncertainty if evidence is weak"]
    if out.get("context_plan") is None:
        out["context_plan"] = []

    return out


def validate_instruction_obj(obj: Dict[str, Any]) -> None:
    missing = REQUIRED_KEYS - set(obj.keys())
    if missing:
        raise ValueError(f"Instruction JSON missing keys: {sorted(missing)}")

    if obj["answer_mode"] not in ("abstractive", "extractive", "refuse"):
        raise ValueError("answer_mode must be one of: abstractive, extractive, refuse")

    if not isinstance(obj["required_sections"], list):
        raise ValueError("required_sections must be a list")

    if not isinstance(obj["constraints"], list):
        raise ValueError("constraints must be a list")

    if not isinstance(obj["context_plan"], list):
        raise ValueError("context_plan must be a list")


# ============================================================
# 7) Instructor runner helpers (sanitize)
# ============================================================

def sanitize_instructor_json(text: str) -> str:
    """
    Heuristically clean common Qwen JSON corruptions so json.loads can succeed.
    Targeted to typical failure modes: leading/trailing junk, double braces, bracket mishaps, etc.
    Also strips any stray clarifying-question artifacts if the model produces them anyway.
    """
    s = text.strip()

    first = s.find("{")
    if first != -1:
        s = s[first:]

    s = re.sub(r"^\s*\{\s*\{", "{", s)

    last = s.rfind("}")
    if last != -1:
        s = s[: last + 1]

    s = re.sub(r"\]\s*\]\s*,", "],", s)
    s = re.sub(r",\s*,", ",", s)

    return s


def parse_instruction_json(raw_text: str) -> Dict[str, Any]:
    candidate = extract_first_json_object(raw_text)
    candidate = sanitize_instructor_json(candidate)

    parsed = json.loads(candidate)
    obj = normalize_instruction_obj(parsed)
    validate_instruction_obj(obj)
    return obj
