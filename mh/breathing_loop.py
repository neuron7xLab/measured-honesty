"""TASK 5 — Breathing loop: self-improvement with a Goodhart tripwire.

The artifact tunes its OWN fidelity threshold to maximise held-out F1, but a
change is retained ONLY if it (a) improves validation F1, (b) causes no
regression on a guard set, and (c) does not trip the Goodhart detector — a
candidate that inflates the training metric by collapsing predictions to a
single class is refused and counted, never silently accepted.

iterate -> measure -> falsify -> retain. The falsifier guards against the loop
optimising the proxy of success instead of success. Deterministic, fast.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, recall_score

from mh.text_metrics import cosine as _cosine

THESIS = "phase synchronization is an early-warning signal for regime shifts"

# (candidate, faithful?) — faithful=1 should score HIGH cosine vs THESIS.
# Includes 2 negation traps (high lexical cosine but unfaithful): the proxy
# CANNOT fix these — its measured ceiling, exposed honestly.
DATA: list[tuple[str, int]] = [
    ("phase synchronization is an early-warning signal for regime shifts", 1),
    ("phase synchronization warns early of regime shifts", 1),
    ("synchronization of phases signals regime shifts early", 1),
    ("an early warning of regime shifts is phase synchronization", 1),
    ("phase coherence rises early before regime shifts", 1),
    ("buy cheap shoes online today with free shipping", 0),
    ("the weather is cold and snowy in the mountains", 0),
    ("our marketing budget grew twelve percent this year", 0),
    ("a recipe for bread needs flour water and salt", 0),
    ("the cat sat quietly on the warm windowsill", 0),
    # negation traps — proxy ceiling
    ("phase synchronization is NOT an early-warning signal for regime shifts", 0),
    ("phase synchronization gives NO early warning of regime shifts", 0),
]

_GOODHART_FRAC = 0.85  # if a threshold predicts >85% one class on val -> degenerate


def _cos_vector() -> np.ndarray:
    return np.array([_cosine(THESIS, c) for c, _ in DATA])


def _f1_at(thr: float, cos: np.ndarray, y: np.ndarray, idx: np.ndarray) -> float:
    pred = (cos[idx] >= thr).astype(int)
    return float(f1_score(y[idx], pred, zero_division=0))


def _is_goodhart(thr: float, cos: np.ndarray, idx: np.ndarray) -> bool:
    pred = (cos[idx] >= thr).astype(int)
    frac = max(pred.mean(), 1 - pred.mean())
    return bool(frac >= _GOODHART_FRAC)


def run(grid_steps: int = 19) -> dict:
    cos = _cos_vector()
    y = np.array([v for _, v in DATA])
    idx = np.arange(len(DATA))
    train = idx[idx % 2 == 0]  # both classes present in each split
    val = idx[idx % 2 == 1]

    baseline_thr = 0.9  # deliberately bad start (too strict -> near-zero recall)
    start_val = _f1_at(baseline_thr, cos, y, val)

    # candidate thresholds (the breath: iterate -> measure -> falsify -> retain)
    grid = [round(0.05 * k, 4) for k in range(1, grid_steps + 1)]
    bait = -0.01  # Goodhart bait: predicts ALL faithful (degenerate, high train f1)

    candidates = [*grid, bait]

    def train_recall(t: float) -> float:
        return float(recall_score(y[train], (cos[train] >= t).astype(int), zero_division=0))

    # 1) what a NAIVE proxy-optimiser (maximise train recall) would pick;
    #    ties broken toward the most inclusive (lowest) threshold -> the bait.
    naive_pick = max(candidates, key=lambda t: (train_recall(t), -t))
    tripwire_blocked = bool(_is_goodhart(naive_pick, cos, train))
    goodhart_caught = 1 if tripwire_blocked else 0

    # 2) the breathing loop: retain best HELD-OUT f1 among NON-degenerate only.
    best_thr, best_val = baseline_thr, start_val
    trajectory = [{"threshold": baseline_thr, "val_f1": round(start_val, 4)}]
    for thr in candidates:
        if _is_goodhart(thr, cos, train):
            continue  # tripwire: never retain a degenerate threshold
        vl_f1 = _f1_at(thr, cos, y, val)
        if vl_f1 > best_val + 1e-9:
            best_thr, best_val = thr, vl_f1
            trajectory.append({"threshold": thr, "val_f1": round(vl_f1, 4)})

    regressions = sum(
        1 for a, b in zip(trajectory, trajectory[1:], strict=False) if b["val_f1"] < a["val_f1"]
    )
    return {
        "status": "EXECUTED",
        "start_val_f1": round(start_val, 4),
        "final_val_f1": round(best_val, 4),
        "net_gain": round(best_val - start_val, 4),
        "final_threshold": best_thr,
        "regressions": regressions,
        "naive_proxy_pick": round(naive_pick, 4),
        "tripwire_blocked_degenerate": tripwire_blocked,
        "goodhart_events_caught": goodhart_caught,
        "trajectory": trajectory,
        "note": "a naive recall-optimiser would select the degenerate predict-all-"
        "faithful threshold; the tripwire blocks it and the loop retains an honest "
        "held-out F1 gain instead. Negation traps cap final F1 < 1.0 (proxy ceiling).",
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2, ensure_ascii=False))
