"""Real semantic fidelity backend (optional, EXECUTED when models present).

Two models, because measurement showed one is not enough:
- bi-encoder cosine (all-MiniLM-L6-v2): graded similarity / drift detection.
- NLI cross-encoder (nli-MiniLM2-L6-H768): entailment vs CONTRADICTION — this is
  what catches negation/meaning-flips that pure cosine is blind to (verified:
  cosine scored a negated thesis 0.97, NLI scores it `contradiction`).

Graceful: if sentence-transformers is absent (e.g. the default system
interpreter), available() is False and callers fall back to the structural
proxy. Install path: a venv with sentence-transformers (see .venv-st).
"""

from __future__ import annotations

from functools import lru_cache

EMBED_MODEL = "all-MiniLM-L6-v2"
NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
NLI_LABELS = ("contradiction", "entailment", "neutral")


def available() -> bool:
    try:
        import sentence_transformers  # noqa: F401

        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def _nli():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(NLI_MODEL)


def semantic_cosine(a: str, b: str) -> float:
    from sentence_transformers import util

    m = _embedder()
    ea = m.encode(a, normalize_embeddings=True)
    eb = m.encode(b, normalize_embeddings=True)
    return float(util.cos_sim(ea, eb)[0][0])


def nli_label(premise: str, hypothesis: str) -> str:
    import numpy as np

    scores = _nli().predict([(premise, hypothesis)])[0]
    return NLI_LABELS[int(np.argmax(scores))]


def nli_scores(premise: str, hypothesis: str) -> dict[str, float]:
    """Softmax probabilities over (contradiction, entailment, neutral)."""
    import numpy as np

    logits = np.asarray(_nli().predict([(premise, hypothesis)])[0], dtype=float)
    e = np.exp(logits - logits.max())
    p = e / e.sum()
    return dict(zip(NLI_LABELS, (float(x) for x in p), strict=True))


def faithful(thesis: str, candidate: str) -> tuple[bool, str, float]:
    """A candidate is faithful iff NLI is not 'contradiction'. Returns
    (faithful, nli_label, cosine)."""
    label = nli_label(thesis, candidate)
    cos = semantic_cosine(thesis, candidate)
    return (label != "contradiction", label, cos)
