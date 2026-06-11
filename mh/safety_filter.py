"""Coarse deterministic safety guard for the §9 blocked-pattern list.

HONEST SCOPE: this is a lexical/regex guard, advisory — not a verified safety
classifier. It catches explicit instances; it has unmeasured precision/recall
and will miss paraphrased violations. It fails LOUD (flags), never silent, and
never claims a text is "safe" — only that no listed pattern was detected.
"""

from __future__ import annotations

import re

# pattern name -> compiled matchers (substring-insensitive)
_RAW: dict[str, list[str]] = {
    "fabricated_social_proof": [
        r"\beveryone knows\b",
        r"\bexperts agree\b",
        r"\bstudies show\b",
        r"\b(thousands|millions) of (people|experts|users) (agree|trust)\b",
    ],
    "manipulative_fear_bait": [
        r"\bact now or\b",
        r"\blast chance\b",
        r"\byou will lose everything\b",
        r"\byou'?ll regret\b",
        r"\bbefore it'?s too late\b",
    ],
    "identity_attack": [
        r"\b(idiots?|morons?|stupid people)\b",
        r"\bthey are subhuman\b",
    ],
    "harassment": [r"\bgo kill yourself\b", r"\bi will find you\b"],
    "impersonation": [
        r"\bi am (elon musk|the president|the ceo of)\b",
        r"\bofficial account of\b(?!.*\b(parody|satire|fan)\b)",
    ],
    "deepfake_claim_without_disclosure": [r"\bdeepfake\b(?!.*\b(disclosure|parody|satire)\b)"],
    "spam_automation": [
        r"!{3,}",
        r"\bbuy now\b",
        r"\bdm me to earn\b",
        r"\bclick here to win\b",
    ],
    "false_expertise_signal": [
        r"\bworld'?s #?1\b",
        r"\bguaranteed results\b",
        r"\bas a leading expert\b",
    ],
    "viral_over_truth": [r"\bguaranteed viral\b", r"\bbreak the internet\b"],
}

BLOCKED_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    name: [re.compile(p, re.IGNORECASE) for p in pats] for name, pats in _RAW.items()
}


def detect_blocked(text: str) -> list[str]:
    """Return the names of blocked patterns detected in `text` (possibly empty)."""
    return sorted(
        name for name, matchers in BLOCKED_PATTERNS.items() if any(m.search(text) for m in matchers)
    )


def repair_suggestion(triggered: list[str]) -> str:
    if not triggered:
        return ""
    fixes = {
        "fabricated_social_proof": "cite a real source or drop the appeal-to-numbers",
        "manipulative_fear_bait": "replace urgency-coercion with a factual stake",
        "identity_attack": "attack the argument, not the person",
        "harassment": "remove threatening/abusive language entirely",
        "impersonation": "do not claim another's identity; disclose authorship",
        "deepfake_claim_without_disclosure": "add an explicit synthetic/parody disclosure",
        "spam_automation": "remove spam CTAs and excess punctuation",
        "false_expertise_signal": "drop unverifiable superiority claims",
        "viral_over_truth": "optimise for the verifiable claim, not virality",
    }
    return "; ".join(fixes.get(t, f"remove {t}") for t in triggered)
