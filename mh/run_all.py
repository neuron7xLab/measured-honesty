"""Aggregate the five nuclear-task results into one evidence report.

T1 (lattice), T3 (transfer), T5 (breathing) always run (pure/deterministic).
T2 (calibration), T4 (scaling) run only when NLI + LLM are available; otherwise
they are reported BLOCKED — never faked.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from mh import (  # noqa: E402  # noqa: E402
    breathing_loop,
    llm_critic,
    semantic_embed,
    transfer_falsifier,
)
from mh import evidence_lattice as L  # noqa: E402

BLOCKED = {"status": "BLOCKED", "reason": "needs AOS_LLM_TESTS=1 + sentence-transformers + ollama"}


def main() -> int:
    models_live = (
        os.environ.get("AOS_LLM_TESTS") == "1"
        and semantic_embed.available()
        and llm_critic.available()
    )
    if models_live:
        from mh import critic_calibration, verification_scaling

        t2 = critic_calibration.evaluate()
        t4 = verification_scaling.evaluate()
    else:
        t2 = t4 = dict(BLOCKED)

    report = {
        "T1_evidence_lattice": {
            "status": "EXECUTED",
            "is_bounded_lattice": L.is_bounded_lattice(),
            "is_distributive": L.is_distributive(),
            "law_conjunction_is_meet": L.conjunction_strength([L.EXECUTED, L.ASSUMPTION])
            == L.ASSUMPTION,
            "law_failclosed_green": (
                not L.admits_green(L.conjunction_strength([L.EXECUTED, L.ASSUMPTION]))
                and L.admits_green(L.conjunction_strength([L.EXECUTED, L.SOURCE]))
            ),
        },
        "T2_critic_calibration": t2,
        "T3_transfer_falsifier": transfer_falsifier.evaluate(),
        "T4_verification_scaling": t4,
        "T5_breathing_loop": breathing_loop.run(),
    }
    out = REPO / "artifacts" / "research_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    t1 = report["T1_evidence_lattice"]
    t3 = report["T3_transfer_falsifier"]
    t5 = report["T5_breathing_loop"]
    t3n = f"{t3['over_claims_caught']}/{t3['domains_audited']}"
    print(
        f"RESEARCH: T1 lattice={t1['is_bounded_lattice']} | "
        f"T3 neuro_null={t3['neuro_null_reproduced']} ({t3n}) | "
        f"T5 gain={t5['net_gain']} goodhart_caught={t5['goodhart_events_caught']} | "
        f"T2/T4 {'EXECUTED' if models_live else 'BLOCKED'} -> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
