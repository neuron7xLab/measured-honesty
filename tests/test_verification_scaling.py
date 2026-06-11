"""TASK 4 proof — verification scaling law (quality vs verifier compute).

Gated on venv (NLI) + ollama (LLM); locks the measured saturation curve.
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
    from mh.verification_scaling import evaluate

    return evaluate()


def test_routed_quality_is_monotone_non_decreasing(report: dict) -> None:
    accs = [c["routed_accuracy"] for c in report["curve"]]
    assert all(b >= a for a, b in zip(accs, accs[1:], strict=False))


def test_routed_saturates_and_max_is_perfect(report: dict) -> None:
    assert report["max_routed_accuracy"] == 1.0
    # saturates before the full stack (the redundant verifier adds nothing)
    assert report["routed_saturates_at_n"] <= len(report["curve"]) - 1


def test_extra_verifier_adds_zero_routed_gain(report: dict) -> None:
    curve = report["curve"]
    assert curve[-1]["routed_accuracy"] == curve[-2]["routed_accuracy"]
