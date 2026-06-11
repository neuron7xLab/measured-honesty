"""Scope-boundary test — §9 guard on real SMS spam. Opt-in (network)."""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("AOS_NET_TESTS") != "1",
    reason="set AOS_NET_TESTS=1 (network) to run",
)


def test_guard_is_not_a_spam_classifier() -> None:
    from mh.benchmark_spam import evaluate

    r = evaluate(n=600, seed=0)
    assert r["status"] == "EXECUTED"
    # the honest boundary: high ham specificity, low spam recall (not a spam tool)
    assert r["ham_specificity"] >= 0.95
    assert r["spam_recall"] <= 0.20
