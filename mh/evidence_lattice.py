"""Evidence lattice — the algebra of verified delegation (constinsence core).

The AOS evidence classes form a BOUNDED LATTICE under epistemic strength, not a
flat list. This module makes that structure first-class and provable:

    EXECUTED                      (top: verified by running)
       |
    SOURCE                        (cited external ground)
      /  \\
 SPECIFIED  INFERENCE             (incomparable: contract vs derivation)
      \\  /
   ASSUMPTION                     (unverified premise)
       |
    UNKNOWN                       (bottom: no information)

Two laws make it a discipline, not a label set:

  1. FAIL-CLOSED GREEN: a critical claim may be GREEN only if its evidence is
     >= SOURCE. Weak classes (SPECIFIED/INFERENCE/ASSUMPTION/UNKNOWN) cannot back
     a critical GREEN. `admits_green`.

  2. CONJUNCTION = MEET: a conclusion built from premises is only as strong as
     the MEET (greatest lower bound) of its critical premises — the weakest leg
     caps the whole. `conjunction_strength`. This is coherence made arithmetic:
     you cannot conclude stronger than your weakest critical support.

RISK and DECISION are orthogonal annotations, not strength levels; they are kept
out of the strength lattice on purpose.
"""

from __future__ import annotations

from functools import reduce

# strength classes (lattice carrier)
UNKNOWN = "UNKNOWN"
ASSUMPTION = "ASSUMPTION"
SPECIFIED = "SPECIFIED"
INFERENCE = "INFERENCE"
SOURCE = "SOURCE"
EXECUTED = "EXECUTED"

CLASSES: tuple[str, ...] = (UNKNOWN, ASSUMPTION, SPECIFIED, INFERENCE, SOURCE, EXECUTED)
TOP = EXECUTED
BOTTOM = UNKNOWN

# covering relations: (lower, upper) — b directly covers a
_COVERS: tuple[tuple[str, str], ...] = (
    (UNKNOWN, ASSUMPTION),
    (ASSUMPTION, SPECIFIED),
    (ASSUMPTION, INFERENCE),
    (SPECIFIED, SOURCE),
    (INFERENCE, SOURCE),
    (SOURCE, EXECUTED),
)


def _reachable() -> dict[str, frozenset[str]]:
    """up[a] = {b : a <= b}, reflexive-transitive closure of _COVERS."""
    up: dict[str, set[str]] = {c: {c} for c in CLASSES}
    changed = True
    while changed:
        changed = False
        for lo, hi in _COVERS:
            if not up[hi] <= up[lo]:
                up[lo] |= up[hi]
                changed = True
    return {c: frozenset(s) for c, s in up.items()}


_UP = _reachable()  # a -> {b : a <= b}
_DOWN = {c: frozenset(x for x in CLASSES if c in _UP[x]) for c in CLASSES}  # a -> {b : b <= a}


def leq(a: str, b: str) -> bool:
    """Partial order: a <= b (a is no stronger than b)."""
    _check(a)
    _check(b)
    return b in _UP[a]


def _check(c: str) -> None:
    if c not in _UP:
        raise ValueError(f"not an evidence class: {c!r}; valid: {CLASSES}")


def join(a: str, b: str) -> str:
    """Least upper bound: the WEAKEST common upper bound (largest up-set)."""
    _check(a)
    _check(b)
    common_up = _UP[a] & _UP[b]
    return max(common_up, key=lambda x: len(_UP[x]))


def meet(a: str, b: str) -> str:
    """Greatest lower bound: the STRONGEST common lower bound (smallest up-set)
    — the conjunction cap (weakest critical leg)."""
    _check(a)
    _check(b)
    common_down = _DOWN[a] & _DOWN[b]
    return min(common_down, key=lambda x: len(_UP[x]))


def is_bounded_lattice() -> bool:
    """Verify carrier is a bounded lattice: top, bottom, and unique join+meet
    for every pair (existence is guaranteed by min/max; this checks the lattice
    axioms hold, i.e. join/meet are genuine LUB/GLB)."""
    for a in CLASSES:
        if not (leq(BOTTOM, a) and leq(a, TOP)):
            return False
        for b in CLASSES:
            j, m = join(a, b), meet(a, b)
            if not (leq(a, j) and leq(b, j) and leq(m, a) and leq(m, b)):
                return False
            # LUB/GLB minimality/maximality
            for c in CLASSES:
                if leq(a, c) and leq(b, c) and not leq(j, c):
                    return False
                if leq(c, a) and leq(c, b) and not leq(c, m):
                    return False
    return True


# --- the two disciplines -------------------------------------------------
def admits_green(evidence_class: str) -> bool:
    """A critical claim may be GREEN only if its evidence is >= SOURCE."""
    return leq(SOURCE, evidence_class)


def conjunction_strength(classes: list[str]) -> str:
    """Strength of a conclusion = MEET of its critical premises (weakest leg).
    Empty conjunction is vacuously the top (no premise constrains it)."""
    if not classes:
        return TOP
    return reduce(meet, classes)


def strength_rank(evidence_class: str) -> int:
    """Total preorder index for display/sorting (not the partial order)."""
    _check(evidence_class)
    return len(_DOWN[evidence_class]) - 1
