from __future__ import annotations

import re

from src.analyzer.signals import WORD_RE

WHITESPACE_RE = re.compile(r"\s+")

FILLER_PREFIXES: tuple[tuple[str, str], ...] = (
    (r"^\s*please\s+", ""),
    (r"^\s*kindly\s+", ""),
    (r"^\s*could you please\s+", ""),
    (r"^\s*could you\s+", ""),
    (r"^\s*can you please\s+", ""),
    (r"^\s*can you\s+", ""),
    (r"^\s*i would like you to\s+", ""),
    (r"^\s*i want you to\s+", ""),
    (r"^\s*i was wondering if you could\s+", ""),
    (r"^\s*i need you to\s+", ""),
)

FILLER_INLINE: tuple[tuple[str, str], ...] = (
    (r"\bplease\b", ""),
    (r"\bkindly\b", ""),
    (r"\bfor me\b", ""),
    (r"\bif you don't mind\b", ""),
    (r"\bwhen you get a chance\b", ""),
)

REDUNDANCY_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (r"\bin order to\b", "to"),
    (r"\bdue to the fact that\b", "because"),
    (r"\bat this point in time\b", "now"),
    (r"\bin the event that\b", "if"),
    (r"\bwith regard to\b", "about"),
    (r"\bfor the purpose of\b", "for"),
)

POLITE_WRAPPERS: tuple[tuple[str, str], ...] = (
    (r"^\s*i am writing to ask you to\s+", ""),
    (r"^\s*i am trying to\s+", ""),
    (r"^\s*help me\s+", ""),
)


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text.strip())


def remove_fillers(text: str, aggressive: bool = False) -> tuple[str, list[str]]:
    changes: list[str] = []
    updated = text

    patterns = list(FILLER_PREFIXES)
    if aggressive:
        patterns.extend(POLITE_WRAPPERS)
        patterns.extend(FILLER_INLINE)

    for pattern, replacement in patterns:
        new_text, count = re.subn(pattern, replacement, updated, flags=re.IGNORECASE)
        if count:
            changes.append(f"Removed filler phrase matching `{pattern}`.")
            updated = new_text

    return normalize_whitespace(updated), changes


def compress_redundancy(text: str) -> tuple[str, list[str]]:
    changes: list[str] = []
    updated = text

    for pattern, replacement in REDUNDANCY_REPLACEMENTS:
        new_text, count = re.subn(pattern, replacement, updated, flags=re.IGNORECASE)
        if count:
            changes.append(f"Compressed redundancy `{pattern}` -> `{replacement}`.")
            updated = new_text

    return normalize_whitespace(updated), changes


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
