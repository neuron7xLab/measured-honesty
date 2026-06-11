"""TASK 3 — Transfer falsifier: is the kernel a general scientific instrument?

Claim under test: the AOS evidence calculus is DOMAIN-NEUTRAL. We try to falsify
it by feeding inferences from unrelated domains (software, neuroscience, finance)
through the SAME generic machinery (evidence_lattice) and asking whether it
independently flags a KNOWN over-claim — concluding "confirmed/EXECUTED" while
resting on an unverified (ASSUMPTION) critical premise.

If the generic kernel reproduces a domain-specific negative (e.g. the structure
of the ds003458 PLV null: a confirmation claimed on an assumed frequency match),
that is transfer-validity at the inference-discipline layer. It does NOT claim to
have run the real EEG — wiring real ds003458 numbers is the next step, BLOCKED on
the data files. What transfers is the LAW, measured here.
"""

from __future__ import annotations

from dataclasses import dataclass

from mh import evidence_lattice as L


@dataclass(frozen=True)
class Premise:
    name: str
    evidence_class: str
    critical: bool = True


@dataclass(frozen=True)
class Inference:
    domain: str
    description: str
    premises: tuple[Premise, ...]
    claimed_conclusion: str  # the evidence class the author CLAIMS for the conclusion


@dataclass(frozen=True)
class AuditResult:
    domain: str
    admissible_strength: str  # what the lattice permits (meet of critical premises)
    claimed: str
    over_claim: bool  # True iff the author claimed stronger than admissible
    admits_green: bool

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "admissible_strength": self.admissible_strength,
            "claimed": self.claimed,
            "over_claim": self.over_claim,
            "admits_green": self.admits_green,
        }


def audit(inf: Inference) -> AuditResult:
    crit = [p.evidence_class for p in inf.premises if p.critical]
    admissible = L.conjunction_strength(crit) if crit else L.TOP
    claimed = inf.claimed_conclusion
    # over-claim iff the claim is NOT admissible (claimed <= admissible). This
    # catches strictly-too-strong AND incomparable claims (a strictly-above-only
    # test missed e.g. INFERENCE claimed when only SPECIFIED is admissible).
    over = not L.is_admissible(claimed, admissible)
    return AuditResult(
        domain=inf.domain,
        admissible_strength=admissible,
        claimed=claimed,
        over_claim=over,
        admits_green=L.admits_green(admissible),
    )


# Cross-domain corpus: same calculus, four unrelated domains.
CORPUS: tuple[Inference, ...] = (
    Inference(
        domain="software",
        description="all premises run and verified -> conclusion verified",
        premises=(
            Premise("tests pass", L.EXECUTED),
            Premise("schema validates", L.EXECUTED),
        ),
        claimed_conclusion=L.EXECUTED,
    ),
    Inference(
        domain="neuroscience (ds003458-shaped null)",
        description="PLV 'confirmed' but the frequency-band match is ASSUMED, not verified",
        premises=(
            Premise("PLV computed on data", L.EXECUTED),
            Premise("oscillator/EEG frequency bands actually match", L.ASSUMPTION),  # the trap
        ),
        claimed_conclusion=L.EXECUTED,  # author over-claims confirmation
    ),
    Inference(
        domain="finance",
        description="backtest run, but i.i.d. stationarity merely assumed",
        premises=(
            Premise("walk-forward backtest executed", L.EXECUTED),
            Premise("regime stationarity holds out-of-sample", L.ASSUMPTION),
        ),
        claimed_conclusion=L.SOURCE,
    ),
    Inference(
        domain="provider capability",
        description="doc-cited capability claimed as live-executed",
        premises=(
            Premise("official docs cite the feature", L.SOURCE),
            Premise("we made a live call", L.UNKNOWN),  # never executed
        ),
        claimed_conclusion=L.EXECUTED,
    ),
    Inference(
        domain="argumentation (incomparable claim)",
        description="claims INFERENCE strength from a SPECIFIED-only premise — "
        "incomparable, missed by a strictly-above test, caught by is_admissible",
        premises=(Premise("a design contract exists", L.SPECIFIED),),
        claimed_conclusion=L.INFERENCE,
    ),
)


def evaluate() -> dict:
    results = [audit(i) for i in CORPUS]
    caught = [r for r in results if r.over_claim]
    # transfer validity: the generic kernel must flag the neuroscience over-claim
    neuro = next(r for r in results if r.domain.startswith("neuroscience"))
    return {
        "status": "EXECUTED",
        "domains_audited": len(results),
        "over_claims_caught": len(caught),
        "neuro_null_reproduced": neuro.over_claim and not neuro.admits_green,
        "results": [r.to_dict() for r in results],
        "note": "one calculus, four domains; the neuroscience over-claim is caught "
        "by the SAME meet law as the software one — inference-discipline transfer",
    }


if __name__ == "__main__":
    import json

    print(json.dumps(evaluate(), indent=2, ensure_ascii=False))
