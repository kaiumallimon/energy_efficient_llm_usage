from __future__ import annotations

import re

WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")
