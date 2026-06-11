"""Deterministic structural text metric (token-set cosine).

Vendored, stdlib-only. A cheap, deterministic similarity used by the fidelity
proxy and the breathing loop. It is NOT semantic (blind to negation by
construction) — that limit is measured, not hidden.
"""

from __future__ import annotations

import math
import re
from collections import Counter


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Zа-яА-ЯіїєґІЇЄҐ0-9]+", text.lower())


def cosine(a: str, b: str) -> float:
    ca, cb = Counter(_tokens(a)), Counter(_tokens(b))
    if not ca or not cb:
        return 0.0
    keys = set(ca) | set(cb)
    dot = sum(ca[k] * cb[k] for k in keys)
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    return dot / (na * nb) if na and nb else 0.0
