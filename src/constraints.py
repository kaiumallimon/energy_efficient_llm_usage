from __future__ import annotations

import re

from src.optimizer.phrases import CONSTRAINT_MARKERS

__all__ = ["extract_constraints"]


def extract_constraints(query: str) -> list[str]:
    constraints: list[str] = []
    for pattern in CONSTRAINT_MARKERS:
        for match in re.finditer(pattern, query, flags=re.IGNORECASE):
            start = max(0, match.start() - 40)
            end = min(len(query), match.end() + 80)
            snippet = query[start:end].strip()
            if snippet and snippet not in constraints:
                constraints.append(snippet)
    return constraints
