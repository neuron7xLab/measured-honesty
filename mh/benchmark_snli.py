"""External-validity check: the NLI critic on real SNLI (open, human-labelled).

Every other gold set in this repo is author-authored (limitation S-5). This is
the one measurement on data nobody here wrote: the Stanford Natural Language
Inference test split (Bowman et al. 2015, CC BY-SA 4.0), pulled live from the
HuggingFace parquet mirror — no local copy, no hand labels.

HONEST SCOPE: the model (`cross-encoder/nli-MiniLM2-L6-H768`) was trained on the
SNLI + MultiNLI family. So this is **in-distribution accuracy on a held-out test
split**, NOT zero-shot transfer. It validates that our NLI wrapper reproduces the
canonical task on canonical data; it is not a claim of novel generalisation.
"""

from __future__ import annotations

import io
import urllib.request

from mh import semantic_embed

PARQUET_URL = (
    "https://huggingface.co/api/datasets/stanfordnlp/snli/parquet/plain_text/test/0.parquet"
)
# SNLI gold: 0=entailment, 1=neutral, 2=contradiction, -1=no consensus (dropped).
MODEL_TO_SNLI = {"entailment": 0, "neutral": 1, "contradiction": 2}


def _load(n: int, seed: int):
    import pandas as pd

    req = urllib.request.Request(PARQUET_URL, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=60).read()  # noqa: S310
    df = pd.read_parquet(io.BytesIO(raw))
    df = df[df.label.isin([0, 1, 2])]
    return df.sample(min(n, len(df)), random_state=seed).reset_index(drop=True)


def evaluate(n: int = 250, seed: int = 0) -> dict:
    """Run the NLI critic on n SNLI test pairs; return accuracy/macro-F1/confusion."""
    if not semantic_embed.available():
        return {"status": "BLOCKED", "reason": "sentence-transformers not importable"}
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    df = _load(n, seed)
    y_true = df.label.tolist()
    y_pred = [
        MODEL_TO_SNLI[semantic_embed.nli_label(p, h)]
        for p, h in zip(df.premise, df.hypothesis, strict=True)
    ]
    return {
        "status": "EXECUTED",
        "dataset": "stanfordnlp/snli (test split, human gold, CC BY-SA 4.0)",
        "n": len(df),
        "seed": seed,
        "model": semantic_embed.NLI_MODEL,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "confusion_true_rows_ent_neu_con": confusion_matrix(
            y_true, y_pred, labels=[0, 1, 2]
        ).tolist(),
        "caveat": "model was trained on the SNLI+MNLI family -> this is "
        "IN-DISTRIBUTION accuracy on a held-out test split, NOT zero-shot transfer. "
        "It is the only measurement here on data not authored in this repo (closes "
        "limitation S-5 for the NLI agent).",
    }


if __name__ == "__main__":
    import json

    print(json.dumps(evaluate(), indent=2, ensure_ascii=False))
