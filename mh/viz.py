"""Deterministic, dependency-free SVG renderer for the evidence lattice.

Two views, both pure-string SVG (no matplotlib, no Date/random — reproducible
byte-for-byte):

  hasse_svg()        the Hasse diagram of the evidence-strength lattice, with
                     the `admits GREEN` frontier drawn as a horizontal rule.
  claim_audit_svg()  a claim graph: critical premises -> meet -> conclusion,
                     coloured by admissibility (the calculus made visible).

Aesthetic: minimal black/white, generous whitespace, one accent for the GREEN
frontier. Top (EXECUTED) is filled; bottom (UNKNOWN) is dashed.
"""

from __future__ import annotations

from mh import evidence_lattice as L

# node centre coordinates (x, y) — a clean modular Hasse layout
_POS: dict[str, tuple[int, int]] = {
    L.EXECUTED: (300, 80),
    L.SOURCE: (300, 200),
    L.SPECIFIED: (160, 320),
    L.INFERENCE: (440, 320),
    L.ASSUMPTION: (300, 440),
    L.UNKNOWN: (300, 560),
}
_W, _H = 600, 640
_NW, _NH = 168, 46  # node box

_INK = "#111111"
_MUT = "#8a8a8a"
_GREEN = "#1a7f37"
_PAPER = "#fafafa"

_FONT = "font-family='ui-sans-serif,-apple-system,Segoe UI,Inter,Roboto,Helvetica,Arial,sans-serif'"


def _node(name: str, cx: int, cy: int) -> str:
    x, y = cx - _NW // 2, cy - _NH // 2
    if name == L.EXECUTED:
        fill, stroke, text, dash = _INK, _INK, "#ffffff", ""
    elif name == L.UNKNOWN:
        fill, stroke, text, dash = "#ffffff", _MUT, _MUT, "stroke-dasharray='5 4'"
    else:
        fill, stroke, text, dash = "#ffffff", _INK, _INK, ""
    rank = L.strength_rank(name)
    return (
        f"<g>"
        f"<rect x='{x}' y='{y}' width='{_NW}' height='{_NH}' rx='9' "
        f"fill='{fill}' stroke='{stroke}' stroke-width='1.6' {dash}/>"
        f"<text x='{cx}' y='{cy + 1}' text-anchor='middle' dominant-baseline='middle' "
        f"{_FONT} font-size='17' font-weight='600' fill='{text}'>{name}</text>"
        f"<text x='{cx + _NW // 2 - 14}' y='{y + 14}' text-anchor='middle' "
        f"{_FONT} font-size='10' fill='{_MUT if name != L.EXECUTED else '#cfcfcf'}'>"
        f"{rank}</text>"
        f"</g>"
    )


def hasse_svg() -> str:
    edges = "".join(
        f"<line x1='{_POS[lo][0]}' y1='{_POS[lo][1] - _NH // 2}' "
        f"x2='{_POS[hi][0]}' y2='{_POS[hi][1] + _NH // 2}' "
        f"stroke='{_INK}' stroke-width='1.3'/>"
        for lo, hi in L._COVERS
    )
    nodes = "".join(_node(n, *_POS[n]) for n in L.CLASSES)
    # the admits-GREEN frontier: a rule between SOURCE and {SPECIFIED, INFERENCE}
    fy = (_POS[L.SOURCE][1] + _POS[L.SPECIFIED][1]) // 2
    frontier = (
        f"<line x1='40' y1='{fy}' x2='{_W - 40}' y2='{fy}' stroke='{_GREEN}' "
        f"stroke-width='1.2' stroke-dasharray='2 5'/>"
        f"<text x='{_W - 46}' y='{fy - 8}' text-anchor='end' {_FONT} font-size='12' "
        f"fill='{_GREEN}'>admits GREEN  ( ≥ SOURCE ) ↑</text>"
    )
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {_W} {_H}' "
        f"width='{_W}' height='{_H}' role='img' "
        f"aria-label='Evidence strength lattice (Hasse diagram)'>"
        f"<rect width='{_W}' height='{_H}' fill='{_PAPER}'/>"
        f"<text x='40' y='44' {_FONT} font-size='20' font-weight='700' fill='{_INK}'>"
        f"Evidence strength — a bounded distributive lattice</text>"
        f"<text x='40' y='{_H - 20}' {_FONT} font-size='12' fill='{_MUT}'>"
        f"conjunction = meet (weakest critical leg caps the conclusion) · "
        f"disjunction = join</text>"
        f"{edges}{frontier}{nodes}</svg>"
    )


def claim_audit_svg(
    premises: list[tuple[str, str]], claimed: str, title: str = "claim audit"
) -> str:
    """premises: [(label, evidence_class)]; renders premises -> meet -> conclusion."""
    classes = [c for _, c in premises]
    admissible = L.conjunction_strength(classes) if classes else L.TOP
    ok = L.is_admissible(claimed, admissible)
    verdict_color = _GREEN if ok else "#b42318"

    w = 640
    top = 84
    row_h = 66
    block = max(1, len(premises)) * row_h
    mid = top + block // 2  # vertical centre of the premise block
    h = top + block + 64  # room for the verdict line below
    rows = []
    for i, (label, cls) in enumerate(premises):
        y = top + i * row_h
        rows.append(
            f"<rect x='30' y='{y}' width='250' height='46' rx='8' fill='#ffffff' "
            f"stroke='{_INK}' stroke-width='1.4'/>"
            f"<text x='44' y='{y + 19}' {_FONT} font-size='13' fill='{_INK}'>{label}</text>"
            f"<text x='44' y='{y + 36}' {_FONT} font-size='12' font-weight='700' "
            f"fill='{_MUT}'>{cls}</text>"
            f"<line x1='280' y1='{y + 23}' x2='350' y2='{mid}' stroke='{_MUT}' "
            f"stroke-width='1.2'/>"
        )
    meet_box = (
        f"<rect x='350' y='{mid - 26}' width='120' height='52' rx='10' fill='#ffffff' "
        f"stroke='{_INK}' stroke-width='1.6'/>"
        f"<text x='410' y='{mid - 4}' text-anchor='middle' {_FONT} font-size='12' "
        f"fill='{_MUT}'>meet =</text>"
        f"<text x='410' y='{mid + 14}' text-anchor='middle' {_FONT} font-size='14' "
        f"font-weight='700' fill='{_INK}'>{admissible}</text>"
        f"<line x1='470' y1='{mid}' x2='520' y2='{mid}' stroke='{_MUT}' "
        f"stroke-width='1.2'/>"
    )
    concl = (
        f"<rect x='520' y='{mid - 28}' width='96' height='56' rx='10' fill='#ffffff' "
        f"stroke='{verdict_color}' stroke-width='2'/>"
        f"<text x='568' y='{mid - 6}' text-anchor='middle' {_FONT} font-size='11' "
        f"fill='{_MUT}'>claimed</text>"
        f"<text x='568' y='{mid + 12}' text-anchor='middle' {_FONT} font-size='13' "
        f"font-weight='700' fill='{verdict_color}'>{claimed}</text>"
    )
    verdict = (
        f"<text x='30' y='{h - 22}' {_FONT} font-size='13' font-weight='700' "
        f"fill='{verdict_color}'>"
        f"{'ADMISSIBLE' if ok else 'OVER-CLAIM'} — claimed {claimed} "
        f"{'≤' if ok else '⋠'} admissible {admissible}</text>"
    )
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' width='{w}' "
        f"height='{h}' role='img' aria-label='{title}'>"
        f"<rect width='{w}' height='{h}' fill='{_PAPER}'/>"
        f"<text x='30' y='40' {_FONT} font-size='18' font-weight='700' fill='{_INK}'>"
        f"{title}</text>"
        f"{''.join(rows)}{meet_box}{concl}{verdict}</svg>"
    )


def main() -> int:
    from pathlib import Path

    docs = Path(__file__).resolve().parent.parent / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "evidence_lattice.svg").write_text(hasse_svg())
    # the ds003458-shaped over-claim, made visible
    audit = claim_audit_svg(
        [("PLV computed on data", L.EXECUTED), ("frequency bands match", L.ASSUMPTION)],
        L.EXECUTED,
        title="claim audit — ds003458-shaped over-claim",
    )
    (docs / "claim_audit_ds003458.svg").write_text(audit)
    print(f"wrote {docs}/evidence_lattice.svg and {docs}/claim_audit_ds003458.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
