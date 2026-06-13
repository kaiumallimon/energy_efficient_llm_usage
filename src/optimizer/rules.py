from __future__ import annotations

import re

from src.optimizer.guardrails import has_constraint_markers, intent_preserved, safe_optimize
from src.optimizer.phrases import (
    DUPLICATE_INTENSIFIER_RE,
    FILLER_INLINE,
    FILLER_PREFIXES,
    FILLER_SUFFIXES,
    GRAMMAR_FIXES,
    HEDGE_PHRASES,
    REDUNDANCY_REPLACEMENTS,
)

INLINE_FILLER_CLAUSES: tuple[str, ...] = (
    r",?\s*if you don't mind\s*,?",
    r",?\s*if that'?s okay\s*,?",
    r",?\s*if possible\s*,?",
    r",?\s*when you get a chance\s*,?",
    r",?\s*whenever you can\s*,?",
    r",?\s*thanks in advance\s*,?",
    r",?\s*thank you in advance\s*,?",
)
from src.text_utils import WORD_RE, WHITESPACE_RE
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
CLAUSE_SPLIT_RE = re.compile(r"\s*[,;]\s*")


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text.strip())


def normalize_punctuation(text: str) -> str:
    updated = text
    for pattern, replacement in GRAMMAR_FIXES:
        updated = re.sub(pattern, replacement, updated, flags=re.IGNORECASE)
    return normalize_whitespace(updated)


def capitalize_first(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped
    return stripped[0].upper() + stripped[1:]


def _apply_patterns(
    text: str,
    patterns: tuple[str, ...],
    *,
    replacement: str = "",
    anchor_start: bool = False,
    anchor_end: bool = False,
) -> tuple[str, list[str]]:
    changes: list[str] = []
    updated = text

    for pattern in patterns:
        full_pattern = pattern
        if anchor_start:
            full_pattern = rf"^\s*{pattern}"
        if anchor_end:
            full_pattern = rf"{pattern}$" if anchor_start else rf"{pattern}\s*$"

        new_text, count = re.subn(full_pattern, replacement, updated, flags=re.IGNORECASE)
        if count:
            changes.append(f"Removed filler matching `{pattern}`.")
            updated = new_text

    return normalize_whitespace(updated), changes


SHORT_GREETING_PREFIXES = frozenset({r"hey,?\s+", r"hi\s+", r"hello\s+", r"so,?\s+"})


def remove_prefix_fillers(text: str) -> tuple[str, list[str]]:
    changes: list[str] = []
    updated = text
    changed = True

    while changed:
        changed = False
        for pattern in FILLER_PREFIXES:
            new_text, count = re.subn(
                rf"^\s*{pattern}",
                "",
                updated,
                count=1,
                flags=re.IGNORECASE,
            )
            if not count:
                continue

            candidate = normalize_whitespace(new_text)
            if pattern in SHORT_GREETING_PREFIXES and count_words(candidate) <= 3:
                continue

            changes.append(f"Removed prefix filler `{pattern}`.")
            updated = candidate
            changed = True
            break

    return updated, changes


def remove_inline_filler_clauses(text: str) -> tuple[str, list[str]]:
    changes: list[str] = []
    updated = text

    for pattern in INLINE_FILLER_CLAUSES:
        new_text, count = re.subn(pattern, "", updated, flags=re.IGNORECASE)
        if count:
            changes.append(f"Removed inline filler clause `{pattern}`.")
            updated = new_text

    updated = re.sub(r",\s*,", ",", updated)
    updated = re.sub(r"\s+,", ",", updated)
    return normalize_whitespace(updated), changes


def remove_suffix_fillers(text: str) -> tuple[str, list[str]]:
    changes: list[str] = []
    updated = text
    changed = True

    while changed:
        changed = False
        for pattern in FILLER_SUFFIXES:
            new_text, count = re.subn(pattern, "", updated, count=1, flags=re.IGNORECASE)
            if count:
                changes.append(f"Removed suffix filler `{pattern}`.")
                updated = normalize_whitespace(new_text)
                changed = True
                break

    return updated, changes


def remove_inline_fillers(text: str, *, include_hedges: bool = False) -> tuple[str, list[str]]:
    changes: list[str] = []
    updated = text
    patterns = list(FILLER_INLINE)
    if include_hedges:
        patterns.extend(HEDGE_PHRASES)

    for pattern in patterns:
        new_text, count = re.subn(pattern, "", updated, flags=re.IGNORECASE)
        if count:
            changes.append(f"Removed inline phrase `{pattern}`.")
            updated = new_text

    return normalize_whitespace(updated), changes


def remove_fillers(text: str, aggressive: bool = False) -> tuple[str, list[str]]:
    changes: list[str] = []
    updated = text

    updated, prefix_changes = remove_prefix_fillers(updated)
    changes.extend(prefix_changes)

    if aggressive:
        updated, suffix_changes = remove_suffix_fillers(updated)
        changes.extend(suffix_changes)

        updated, inline_clause_changes = remove_inline_filler_clauses(updated)
        changes.extend(inline_clause_changes)

        updated, inline_changes = remove_inline_fillers(updated, include_hedges=True)
        changes.extend(inline_changes)

    return updated, changes


def compress_redundancy(text: str) -> tuple[str, list[str]]:
    changes: list[str] = []
    updated = text

    for pattern, replacement in REDUNDANCY_REPLACEMENTS:
        new_text, count = re.subn(pattern, replacement, updated, flags=re.IGNORECASE)
        if count:
            changes.append(f"Compressed redundancy `{pattern}` -> `{replacement}`.")
            updated = new_text

    new_text, count = DUPLICATE_INTENSIFIER_RE.subn(r"\1", updated)
    if count:
        changes.append("Collapsed duplicate intensifiers.")
        updated = new_text

    return normalize_whitespace(updated), changes


def dedupe_sentences(text: str) -> tuple[str, list[str]]:
    parts = SENTENCE_SPLIT_RE.split(normalize_whitespace(text))
    if len(parts) <= 1:
        return text, []

    seen: set[str] = set()
    unique: list[str] = []
    for part in parts:
        key = part.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(part.strip())

    if len(unique) == len(parts):
        return text, []

    return " ".join(unique), ["Removed duplicate sentences."]


def dedupe_clauses(text: str) -> tuple[str, list[str]]:
    parts = CLAUSE_SPLIT_RE.split(normalize_whitespace(text))
    if len(parts) <= 1:
        return text, []

    seen: set[str] = set()
    unique: list[str] = []
    for part in parts:
        key = part.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(part.strip())

    if len(unique) == len(parts):
        return text, []

    return ", ".join(unique), ["Removed duplicate clauses."]


def polish_query(text: str) -> tuple[str, list[str]]:
    changes: list[str] = []
    updated = normalize_punctuation(text)
    if updated != text:
        changes.append("Normalized punctuation and grammar.")

    capitalized = capitalize_first(updated)
    if capitalized != updated:
        changes.append("Capitalized sentence start.")
        updated = capitalized

    return updated, changes


def trim_context(text: str, max_words: int) -> tuple[str, list[str]]:
    words = WORD_RE.findall(text)
    if len(words) <= max_words:
        return text, []

    trimmed = " ".join(words[:max_words])
    changes = [f"Trimmed context from {len(words)} to {max_words} words."]
    return trimmed, changes


def dedupe_lines(text: str) -> tuple[str, list[str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) <= 1:
        return text, []

    seen: set[str] = set()
    unique_lines: list[str] = []
    for line in lines:
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        unique_lines.append(line)

    if len(unique_lines) == len(lines):
        return text, []

    return "\n".join(unique_lines), ["Removed duplicate context lines."]


def optimize_query_text(text: str, *, aggressive: bool, moderate: bool) -> tuple[str, list[str]]:
    """Multi-pass optimization pipeline for raw user queries."""
    original = text.strip()
    if not original:
        return original, []

    changes: list[str] = []
    updated = original

    normalized = normalize_whitespace(updated)
    if normalized != updated:
        changes.append("Normalized query whitespace.")
        updated = normalized

    if aggressive or moderate:
        def filler_step(value: str) -> tuple[str, list[str]]:
            return remove_fillers(value, aggressive=aggressive)

        updated, filler_changes = safe_optimize(updated, filler_step)
        changes.extend(filler_changes)

        if aggressive:
            def redundancy_step(value: str) -> tuple[str, list[str]]:
                return compress_redundancy(value)

            updated, redundancy_changes = safe_optimize(updated, redundancy_step)
            changes.extend(redundancy_changes)

            updated, sentence_changes = dedupe_sentences(updated)
            changes.extend(sentence_changes)

            if not has_constraint_markers(original):
                updated, clause_changes = dedupe_clauses(updated)
                changes.extend(clause_changes)

        if moderate and not aggressive:
            updated, inline_changes = remove_inline_fillers(updated, include_hedges=False)
            changes.extend(inline_changes)

    updated, polish_changes = polish_query(updated)
    changes.extend(polish_changes)

    if not intent_preserved(original, updated):
        return original, ["Rolled back optimization to preserve user intent."]

    return updated, changes
