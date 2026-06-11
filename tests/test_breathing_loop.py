"""TASK 5 proof — self-improvement with a working Goodhart tripwire.

Pure (deterministic cosine + sklearn), runs in the default gate.
"""

from mh.breathing_loop import run


def test_real_held_out_gain_with_no_regression() -> None:
    r = run()
    assert r["net_gain"] > 0.0
    assert r["regressions"] == 0


def test_tripwire_blocks_the_naive_proxy_degenerate() -> None:
    r = run()
    # a naive recall-optimiser would pick the degenerate threshold; tripwire stops it
    assert r["tripwire_blocked_degenerate"] is True
    assert r["goodhart_events_caught"] >= 1


def test_proxy_ceiling_is_honest() -> None:
    # negation traps cap held-out F1 strictly below 1.0 — no fake perfection
    assert run()["final_val_f1"] < 1.0
