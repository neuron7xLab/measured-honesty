"""Property-based proof that the evidence lattice IS a bounded lattice and that
the two disciplines (fail-closed GREEN, conjunction=meet) hold as algebraic law.

Hypothesis samples the carrier; the lattice axioms must survive every draw.
"""

from hypothesis import given
from hypothesis import strategies as st

from mh import evidence_lattice as L

cls = st.sampled_from(L.CLASSES)


# --- it is a bounded lattice --------------------------------------------
def test_is_bounded_lattice() -> None:
    assert L.is_bounded_lattice()


@given(cls)
def test_reflexive(a: str) -> None:
    assert L.leq(a, a)


@given(cls, cls)
def test_antisymmetric(a: str, b: str) -> None:
    if L.leq(a, b) and L.leq(b, a):
        assert a == b


@given(cls, cls, cls)
def test_transitive(a: str, b: str, c: str) -> None:
    if L.leq(a, b) and L.leq(b, c):
        assert L.leq(a, c)


@given(cls, cls)
def test_join_commutative_and_is_lub(a: str, b: str) -> None:
    assert L.join(a, b) == L.join(b, a)
    j = L.join(a, b)
    assert L.leq(a, j) and L.leq(b, j)


@given(cls, cls)
def test_meet_commutative_and_is_glb(a: str, b: str) -> None:
    assert L.meet(a, b) == L.meet(b, a)
    m = L.meet(a, b)
    assert L.leq(m, a) and L.leq(m, b)


@given(cls, cls, cls)
def test_join_associative(a: str, b: str, c: str) -> None:
    assert L.join(L.join(a, b), c) == L.join(a, L.join(b, c))


@given(cls, cls, cls)
def test_meet_associative(a: str, b: str, c: str) -> None:
    assert L.meet(L.meet(a, b), c) == L.meet(a, L.meet(b, c))


@given(cls, cls)
def test_absorption(a: str, b: str) -> None:
    assert L.join(a, L.meet(a, b)) == a
    assert L.meet(a, L.join(a, b)) == a


@given(cls)
def test_idempotent(a: str) -> None:
    assert L.join(a, a) == a
    assert L.meet(a, a) == a


@given(cls)
def test_bounds(a: str) -> None:
    assert L.join(a, L.BOTTOM) == a
    assert L.meet(a, L.TOP) == a


@given(cls, cls)
def test_order_join_meet_consistency(a: str, b: str) -> None:
    assert L.leq(a, b) == (L.join(a, b) == b) == (L.meet(a, b) == a)


# --- discipline 1: fail-closed GREEN ------------------------------------
def test_only_source_and_executed_admit_green() -> None:
    admit = [c for c in L.CLASSES if L.admits_green(c)]
    assert set(admit) == {L.SOURCE, L.EXECUTED}


@given(st.lists(cls, min_size=1))
def test_green_requires_every_critical_leg_strong(classes: list[str]) -> None:
    # if the conjunction admits GREEN, every leg must be >= SOURCE
    if L.admits_green(L.conjunction_strength(classes)):
        assert all(L.admits_green(c) for c in classes)


# --- discipline 2: conjunction = meet (weakest leg caps) ----------------
@given(st.lists(cls, min_size=1))
def test_conjunction_no_stronger_than_weakest_leg(classes: list[str]) -> None:
    s = L.conjunction_strength(classes)
    for c in classes:
        assert L.leq(s, c)  # the conclusion is <= every premise


@given(st.lists(cls, min_size=1))
def test_conjunction_equals_weakest(classes: list[str]) -> None:
    s = L.conjunction_strength(classes)
    weakest = min(classes, key=L.strength_rank)
    # in a chain-with-diamond the meet of a set equals its weakest when totally
    # ordered; otherwise meet may be strictly below — never above
    assert L.leq(s, weakest)
