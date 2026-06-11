"""TASK 3 proof — the generic evidence calculus transfers across domains.

Pure (lattice only), runs in the default gate.
"""

from mh.transfer_falsifier import CORPUS, audit, evaluate


def test_software_all_executed_is_not_overclaim() -> None:
    soft = next(i for i in CORPUS if i.domain == "software")
    assert audit(soft).over_claim is False


def test_neuro_null_reproduced_by_generic_kernel() -> None:
    r = evaluate()
    # the ds003458-shaped over-claim (confirmation on an ASSUMPTION) is caught
    assert r["neuro_null_reproduced"] is True


def test_four_of_five_overclaims_caught() -> None:
    r = evaluate()
    assert r["over_claims_caught"] == 4
    assert r["domains_audited"] == 5


def test_overclaim_uses_the_meet_law() -> None:
    # finance: backtest EXECUTED but stationarity ASSUMPTION -> meet = ASSUMPTION
    fin = next(i for i in CORPUS if i.domain == "finance")
    a = audit(fin)
    assert a.admissible_strength == "ASSUMPTION"
    assert a.over_claim is True


def test_incomparable_overclaim_is_caught() -> None:
    # the gap a strictly-above test missed: INFERENCE claimed from SPECIFIED-only
    arg = next(i for i in CORPUS if i.domain.startswith("argumentation"))
    a = audit(arg)
    assert a.admissible_strength == "SPECIFIED"
    assert a.claimed == "INFERENCE"
    assert a.over_claim is True  # incomparable, not strictly-above
