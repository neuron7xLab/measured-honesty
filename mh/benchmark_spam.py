"""Scope-boundary measurement: the §9 safety guard on real SMS spam.

This is the deliberately *unflattering* counterpart to the SNLI check. The
keyword safety guard targets marketing-MANIPULATION (fabricated social proof,
fear-bait, impersonation, viral hype). It is NOT a spam classifier. Running it on
the UCI SMS Spam Collection (open, CC BY 4.0, via the HF parquet mirror) measures
exactly that boundary: high specificity on ham, near-zero recall on generic spam,
because "you won a prize, call now" is not a §9 manipulation pattern.

A measured negative is a product: this empirically pins the guard's scope instead
of letting anyone assume it catches spam.
"""
from __future__ import annotations

import io
import urllib.request

from mh.safety_filter import detect_blocked

PARQUET_URL = (
    "https://huggingface.co/api/datasets/ucirvine/sms_spam/parquet/"
    "plain_text/train/0.parquet"
)


def _load(n: int, seed: int):
    import pandas as pd

    req = urllib.request.Request(PARQUET_URL, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=60).read()  # noqa: S310
    df = pd.read_parquet(io.BytesIO(raw))
    return df.sample(min(n, len(df)), random_state=seed).reset_index(drop=True)


def evaluate(n: int = 800, seed: int = 0) -> dict:
    """Run the §9 guard on n SMS messages; report scope-boundary characteristics."""
    from sklearn.metrics import accuracy_score, confusion_matrix

    df = _load(n, seed)
    y = df["label"].astype(int).tolist()  # 1 = spam, 0 = ham
    yp = [1 if detect_blocked(str(t)) else 0 for t in df["sms"]]
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "status": "EXECUTED",
        "dataset": "ucirvine/sms_spam (open, CC BY 4.0)",
        "n": len(df),
        "seed": seed,
        "spam_recall": round(recall, 4),
        "ham_specificity": round(specificity, 4),
        "accuracy": round(float(accuracy_score(y, yp)), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "interpretation": "SCOPE BOUNDARY, not a failure. The §9 guard targets "
        "marketing-manipulation, not generic spam; near-zero spam recall with high "
        "ham specificity is the expected, honest result. Do NOT use this guard as a "
        "spam classifier.",
    }


if __name__ == "__main__":
    import json

    print(json.dumps(evaluate(), indent=2, ensure_ascii=False))
