from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


DOSE_REGEX = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:(?:units|mcg|µg|mg|ml|iu|g)\b|%)",
    re.IGNORECASE,
)
FREQ_REGEX = re.compile(r"\b(once|twice|thrice)\s+(daily|a day)|\b(q\d+h|qd|bid|tid|qid)\b", re.IGNORECASE)
SAFETY_REGEX = re.compile(
    r"\b(contraindicated|adverse|toxicity|overdose|interaction|avoid|warning|monitor)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RiskTermDetection:
    flags: List[str]
    matches: List[str]


def detect_high_risk_terms(text: str) -> RiskTermDetection:
    """Detect high-risk patterns/terms in free text.

    Mirrors the notebook’s regex-based high-risk routing signals:
    - dose expressions
    - frequency shorthand
    - safety keywords
    """

    flags: List[str] = []
    matches: List[str] = []

    if text is None:
        return RiskTermDetection(flags=[], matches=[])

    t = str(text)

    dose_matches = [m.group(0) for m in DOSE_REGEX.finditer(t)]
    if dose_matches:
        flags.append("DOSE")
        matches.extend(dose_matches)

    freq_matches = [m.group(0) for m in FREQ_REGEX.finditer(t)]
    if freq_matches:
        flags.append("FREQUENCY")
        matches.extend(freq_matches)

    safety_matches = [m.group(0) for m in SAFETY_REGEX.finditer(t)]
    if safety_matches:
        flags.append("SAFETY")
        matches.extend(safety_matches)

    # de-dup, keep order
    seen = set()
    matches_dedup: List[str] = []
    for m in matches:
        key = m.lower()
        if key not in seen:
            seen.add(key)
            matches_dedup.append(m)

    return RiskTermDetection(flags=flags, matches=matches_dedup)
