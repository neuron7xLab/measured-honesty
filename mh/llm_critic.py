"""Real LLM critic layer (local ollama) — the free-text rhetorical judgement
the deterministic gates and NLI cannot give.

EXECUTED when a local ollama chat model is up; otherwise available() is False
and the MRAS critic uses only its deterministic + NLI gates. No API key, no
cost, fully local. Output is forced to a strict JSON shape and validated.
"""

from __future__ import annotations

from mh import ollama_client

VERDICTS = ("approve", "regenerate", "reject")

# Calibrated against qwen2.5:3b: a small model over-flags without few-shot
# anchoring and an explicit "reworded restatement is faithful" rule. Verified on
# faithful/manipulative/off-thesis/negation cases before deployment.
_SYSTEM = (
    "You are a precise content critic. Decide if CANDIDATE faithfully and safely "
    "represents THESIS. A reworded restatement that PRESERVES the meaning is "
    "faithful and approved, even if wording differs. Reject only deception/"
    "manipulation. Regenerate only if the meaning is unrelated or contradicted. "
    "Reply ONLY with compact JSON."
)

_FEWSHOT = (
    "THESIS: gravity pulls objects toward earth\n"
    "CANDIDATE: things fall down because of gravity\n"
    'JSON: {"verdict":"approve","faithful":true,"manipulation":false,'
    '"reason":"meaning preserved, reworded"}\n\n'
    "THESIS: coffee can improve alertness\n"
    "CANDIDATE: a coffee may help you feel more alert\n"
    'JSON: {"verdict":"approve","faithful":true,"manipulation":false,'
    '"reason":"faithful paraphrase"}\n\n'
    "THESIS: our app improves focus\n"
    "CANDIDATE: GUARANTEED #1, experts agree, buy now or lose everything!!!\n"
    'JSON: {"verdict":"reject","faithful":false,"manipulation":true,'
    '"reason":"fabricated authority, fear-bait"}\n\n'
    "THESIS: bees pollinate flowers\nCANDIDATE: the stock market rose today\n"
    'JSON: {"verdict":"regenerate","faithful":false,"manipulation":false,'
    '"reason":"unrelated to thesis"}\n\n'
)


def available() -> bool:
    return ollama_client.available() and ollama_client.model_present()


def _prompt(thesis: str, candidate: str, register: str) -> str:
    return _FEWSHOT + f"THESIS: {thesis}\nCANDIDATE ({register}): {candidate}\nJSON: "


def critique(thesis: str, candidate: str, register: str = "neutral") -> dict:
    """Return a normalised verdict dict. Fail-closed on malformed model output."""
    try:
        raw = ollama_client.generate_json(_prompt(thesis, candidate, register), system=_SYSTEM)
    except Exception as exc:  # noqa: BLE001 - any model/transport fault -> safe default
        return {
            "verdict": "regenerate",
            "faithful": False,
            "manipulation": False,
            "reason": f"llm_unavailable_or_malformed: {type(exc).__name__}",
            "evidence_class": "RISK",
        }

    verdict = str(raw.get("verdict", "regenerate")).lower()
    if verdict not in VERDICTS:
        verdict = "regenerate"
    return {
        "verdict": verdict,
        "faithful": bool(raw.get("faithful", False)),
        "manipulation": bool(raw.get("manipulation", False)),
        "reason": str(raw.get("reason", ""))[:200],
        "evidence_class": "EXECUTED",
    }
