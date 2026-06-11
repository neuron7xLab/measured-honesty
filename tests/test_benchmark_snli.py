"""External-validity test — NLI critic on real SNLI. Opt-in (network + stack)."""

import os

import pytest

from mh import semantic_embed

pytestmark = pytest.mark.skipif(
    os.environ.get("AOS_NET_TESTS") != "1" or not semantic_embed.available(),
    reason="set AOS_NET_TESTS=1 with sentence-transformers (network) to run",
)


def test_nli_in_distribution_accuracy_strong() -> None:
    from mh.benchmark_snli import evaluate

    r = evaluate(n=200, seed=0)
    assert r["status"] == "EXECUTED"
    assert r["n"] == 200
    # an NLI model on held-out SNLI should be well above chance (0.33)
    assert r["accuracy"] >= 0.80
