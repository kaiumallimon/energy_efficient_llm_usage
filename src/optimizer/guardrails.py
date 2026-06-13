from __future__ import annotations

import re

from src.optimizer.phrases import CONSTRAINT_MARKERS
from src.text_utils import WORD_RE

STOPWORDS = frozenset(
    """
    a an the and or but if then else when at by for with about against between
    into through during before after above below to from up down in out on off
    over under again further once here there all each few more most other some
    such no nor not only own same so than too very can will just don should now
    is are was were be been being have has had do does did of
    """.split()
)

FILLER_WORDS = frozenset(
    """
    could can would will please kindly tell help thanks thank hey hi hello
    wondering appreciate mind maybe perhaps probably actually basically literally
    really just guess mean know want need like look seems feel believe
    you me us your my mine our yours theirs someone somebody anyone
    """.split()
)


def extract_content_words(text: str) -> set[str]:
    words = {w.lower() for w in WORD_RE.findall(text)}
    return {w for w in words if w not in STOPWORDS and len(w) > 1}


def extract_substantive_words(text: str) -> set[str]:
    return {word for word in extract_content_words(text) if word not in FILLER_WORDS}


def has_constraint_markers(text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in CONSTRAINT_MARKERS)


def intent_preserved(original: str, optimized: str, *, min_retention_ratio: float = 0.6) -> bool:
    """Return True when optimized text still contains the core content words."""
    if not optimized.strip():
        return False

    original_words = extract_substantive_words(original)
    if not original_words:
        return bool(optimized.strip())

    optimized_words = extract_content_words(optimized)
    retained = original_words & optimized_words
    ratio = len(retained) / len(original_words)
    return ratio >= min_retention_ratio


def safe_optimize(
    original: str,
    transform,
    *,
    min_retention_ratio: float = 0.6,
) -> tuple[str, list[str]]:
    """Apply a transform and roll back if too much intent is lost."""
    updated, changes = transform(original)
    if intent_preserved(original, updated, min_retention_ratio=min_retention_ratio):
        return updated, changes
    return original, []
