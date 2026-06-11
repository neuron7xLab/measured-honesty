"""TASK viz — deterministic, well-formed SVG of the lattice and claim audits."""

import xml.dom.minidom as minidom

from mh import evidence_lattice as L
from mh import viz


def _wellformed(svg: str) -> None:
    minidom.parseString(svg)  # raises on malformed XML


def test_hasse_is_wellformed_and_deterministic() -> None:
    a, b = viz.hasse_svg(), viz.hasse_svg()
    _wellformed(a)
    assert a == b  # byte-identical: no Date/random
    for cls in L.CLASSES:
        assert cls in a
    assert "admits GREEN" in a


def test_claim_audit_flags_overclaim() -> None:
    svg = viz.claim_audit_svg([("PLV", L.EXECUTED), ("freq match", L.ASSUMPTION)], L.EXECUTED)
    _wellformed(svg)
    assert "OVER-CLAIM" in svg
    assert "ASSUMPTION" in svg  # the meet


def test_claim_audit_passes_admissible() -> None:
    svg = viz.claim_audit_svg([("tests", L.EXECUTED), ("schema", L.EXECUTED)], L.EXECUTED)
    _wellformed(svg)
    assert "ADMISSIBLE" in svg
