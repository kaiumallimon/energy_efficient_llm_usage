"""Catalog of redundant phrase patterns for prompt optimization."""

from __future__ import annotations

import re

# Generic polite openers — safe to strip without removing the user's task verb.
GENERIC_OPENERS: tuple[str, ...] = (
    r"i was just wondering if you could please\s+",
    r"i was wondering if you could please\s+",
    r"i was wondering if you could\s+",
    r"i was wondering if you would\s+",
    r"i would really appreciate it if you could\s+",
    r"i would appreciate it if you could\s+",
    r"i would like to know if you could\s+",
    r"would you be so kind as to\s+",
    r"would you mind helping me\s+",
    r"would you mind telling me\s+",
    r"do you think you could please\s+",
    r"do you think you could\s+",
    r"i am writing to ask you to\s+",
    r"i am trying to figure out\s+",
    r"i need some help to\s+",
    r"i need help figuring out\s+",
    r"i would like you to\s+",
    r"i want you to\s+",
    r"i need you to\s+",
    r"i want to know\s+",
    r"i would like to know\s+",
    r"i am looking for\s+",
    r"i'm looking for\s+",
    r"could you please\s+",
    r"could you\s+",
    r"can you please\s+",
    r"can you\s+",
    r"would you please\s+",
    r"would you\s+",
    r"will you please\s+",
    r"will you\s+",
    r"kindly\s+",
    r"please\s+",
    r"hey,?\s+",
    r"so,?\s+",
)

# Framing phrases that keep the underlying request intact.
REQUEST_FRAMES: tuple[str, ...] = (
    r"help me to\s+",
    r"help me\s+",
    r"please help me\s+",
    r"please tell me,?\s*",
    r"please give me,?\s*",
    r"please show me,?\s*",
    r"could you please tell me,?\s*",
    r"could you please help me,?\s*",
    r"can you please tell me,?\s*",
    r"can you please help me,?\s*",
    r"can you please give me,?\s*",
    r"can you please show me,?\s*",
    r"tell me,?\s*",
    r"i need help to\s+",
    r"i need help with\s+",
)

FILLER_PREFIXES: tuple[str, ...] = GENERIC_OPENERS + REQUEST_FRAMES

FILLER_SUFFIXES: tuple[str, ...] = (
    r"\s+if you don't mind\??",
    r"\s+if that'?s okay\??",
    r"\s+if possible\??",
    r"\s+when you get a chance\??",
    r"\s+whenever you can\??",
    r"\s+thanks in advance\??",
    r"\s+thank you in advance\??",
    r"\s+thank you very much\??",
    r"\s+thanks a lot\??",
    r"\s+thank you\??",
    r"\s+thanks\??",
)

FILLER_INLINE: tuple[str, ...] = (
    r"\bfor me\b",
    r"\bfor us\b",
    r"\bat the moment\b",
    r"\bright now\b",
    r"\bjust\b",
    r"\breally\b",
    r"\bactually\b",
    r"\bbasically\b",
    r"\bliterally\b",
    r"\bkind of\b",
    r"\bsort of\b",
    r"\bmore or less\b",
    r"\bi guess\b",
    r"\bi mean\b",
    r"\byou know\b",
    r"\bif you don't mind\b",
    r"\bwhen you get a chance\b",
)

HEDGE_PHRASES: tuple[str, ...] = (
    r"\bi think\b",
    r"\bi believe\b",
    r"\bi feel like\b",
    r"\bit seems like\b",
    r"\bit looks like\b",
    r"\bperhaps\b",
    r"\bmaybe\b",
    r"\bpossibly\b",
    r"\bprobably\b",
)

REDUNDANCY_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (r"\bin order to understand\b", "to understand"),
    (r"\bin order to\b", "to"),
    (r"\bdue to the fact that\b", "because"),
    (r"\bat this point in time\b", "now"),
    (r"\bin the event that\b", "if"),
    (r"\bwith regard to\b", "about"),
    (r"\bwith respect to\b", "about"),
    (r"\bfor the purpose of\b", "for"),
    (r"\ba majority of\b", "most"),
    (r"\bthe reason is because\b", "because"),
    (r"\brepeat again\b", "repeat"),
    (r"\bcombine together\b", "combine"),
    (r"\bfinal outcome\b", "outcome"),
    (r"\bbasic fundamentals\b", "fundamentals"),
    (r"\bpast history\b", "history"),
    (r"\bend result\b", "result"),
    (r"\bcalculate what is the answer of\b", "what is"),
    (r"\bwhat is the answer of\b", "what is"),
    (r"\bwhat's the answer of\b", "what is"),
    (r"\bcalculate what is\b", "what is"),
    (r"\btell me what is\b", "what is"),
    (r"\bexplain to me what is\b", "what is"),
    (r"\blet me know what is\b", "what is"),
    (r"\bfind out what is\b", "what is"),
)

GRAMMAR_FIXES: tuple[tuple[str, str], ...] = (
    (r"\bwhat the (.+?) is\??\s*$", r"what is the \1?"),
    (r"\bwhat is (.+?) is\??\s*$", r"what is \1?"),
    (r"\bwhat are (.+?) are\??\s*$", r"what are \1?"),
    (r"\?{2,}", "?"),
    (r"!{2,}", "!"),
)

ANALYZER_FILLER_PATTERNS: tuple[str, ...] = (
    r"\b(please|kindly|thanks|thank you)\b",
    r"\b(could you|can you|would you|will you)\b",
    r"\b(i would like|i want|i need)( you)? to\b",
    r"\b(i was wondering|i am wondering)\b",
    r"\b(help me|tell me|let me know)\b",
    r"\b(just|really|actually|basically|literally)\b",
    r"\b(in order to|due to the fact that)\b",
)

DUPLICATE_INTENSIFIER_RE = re.compile(
    r"\b(very|really|extremely|super|quite)\s+(very|really|extremely|super|quite)\b",
    re.IGNORECASE,
)

CONSTRAINT_MARKERS: tuple[str, ...] = (
    r"\bdo not\b",
    r"\bdon't\b",
    r"\bnever\b",
    r"\bmust\b",
    r"\bjson only\b",
    r"\breturn only\b",
    r"\brespond with\b",
    r"\boutput format\b",
)

GREETING_PREFIXES: tuple[str, ...] = (
    r"hey\s+",
    r"hi\s+",
    r"hello\s+",
    r"so\s+",
    r"um\s+",
    r"uh\s+",
)
