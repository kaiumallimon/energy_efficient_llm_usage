from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    similarity: float
    threshold: float
    original_query: str
    candidate_query: str
    accepted_query: str
    source: str
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "similarity": round(self.similarity, 4),
            "threshold": self.threshold,
            "original_query": self.original_query,
            "candidate_query": self.candidate_query,
            "accepted_query": self.accepted_query,
            "source": self.source,
            "notes": self.notes,
        }
