"""TASK 2 proof — measured critic calibration manifold.

Gated on venv (NLI) + ollama (LLM); locks the measured operating point and the
non-trivial finding (naive ensemble hurts; routing recovers).
"""

import os

import pytest

from mh import llm_critic, semantic_embed

pytestmark = pytest.mark.skipif(
    os.environ.get("AOS_LLM_TESTS") != "1"
    or not (semantic_embed.available() and llm_critic.available()),
    reason="needs AOS_LLM_TESTS=1 with sentence-transformers + ollama",
)


@pytest.fixture(scope="module")
def report() -> dict:
    from mh.critic_calibration import evaluate

    return evaluate()


def test_nli_and_llm_are_well_calibrated(report: dict) -> None:
    assert report["per_agent"]["nli"]["brier"] < 0.15
    assert report["per_agent"]["llm"]["brier"] < 0.15


def test_red_agent_is_an_overflagger(report: dict) -> None:
    # adversarial agent: high recall, poor precision (flags almost everything)
    assert report["per_agent"]["red"]["precision"] <= 0.6


def test_naive_ensemble_does_not_beat_best_single(report: dict) -> None:
    # the measured finding: a miscalibrated rater sinks the naive mean
    assert report["naive_lift_over_best_single"] <= 0.0


def test_calibration_routing_recovers_best(report: dict) -> None:
    assert report["routed_lift_over_best_single"] >= 0.0
    assert report["calibration_routed"]["selected_agent"] in report["agents"]
