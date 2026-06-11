# measured-honesty

[![ci](https://github.com/neuron7xLab/measured-honesty/actions/workflows/ci.yml/badge.svg)](https://github.com/neuron7xLab/measured-honesty/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Five falsification studies of an instrument's own limits.**
Every figure below is produced by an executed measurement in this repo and is
committed in [`reports/`](reports/). Claims that cannot be measured are marked
`BLOCKED` / `UNKNOWN`. Sample sizes are small and stated; numbers carry
confidence intervals. A full self-audit is in [`AUDIT.md`](AUDIT.md).

```
make verify        # ruff + mypy + pure tests (T1/T3/T5) + report   (no GPU)
make report        # writes artifacts/research_report.json
make viz           # regenerate the lattice diagrams (deterministic SVG)
```

<p align="center"><img src="docs/evidence_lattice.svg" alt="evidence strength lattice" width="440"></p>

The crown is **T1**: evidence strength is a bounded **distributive lattice**, so
"verified delegation" becomes algebra. Formal note with theorems and proofs:
**[`LATTICE.md`](LATTICE.md)**.

---

## Why this exists

Most tools publish what they can do. This one publishes **where it breaks**,
**how wide the error bars are**, and **how well-calibrated it is about its own
judgements** — measured, not asserted. The thesis: an instrument earns trust by
mapping its own failure manifold and refusing to over-state precision.

### First-principle framing (isomorphisms, not claims)

Analogies that motivated the design; **not** empirical assertions:

- **Active Inference** — a verdict is only as strong as the precision of its
  weakest critical premise. T1 makes that a literal algebra (`conjunction = meet`).
- **Attention Schema Theory** — a system that models its own attention is more
  controllable. T2 builds a *schema of the critic itself*: its calibration is the
  system's model of how much to trust its own attention.
- **Goodhart / free-energy** — optimising a proxy of success is not success. T5
  installs a tripwire that refuses a proxy win that collapses the prediction.

---

## 1. Boundary capabilities (measured)

### T1 — Evidence lattice (`mh/evidence_lattice.py`)
Evidence classes form a **bounded, distributive lattice** under epistemic
strength (`UNKNOWN < ASSUMPTION < {SPECIFIED, INFERENCE} < SOURCE < EXECUTED`).
Proven by **21 tests (16 Hypothesis property tests)**:

- `admits_green(e)` ⇔ `e ≥ SOURCE` — a critical claim cannot be GREEN on a weak
  premise.
- `conjunction_strength = meet` — a conclusion is no stronger than its weakest
  critical leg. `conj(EXECUTED, ASSUMPTION) = ASSUMPTION` → not GREEN.
- `disjunction_strength = join` — alternative evidence paths combine by best-path.
- `is_admissible(claimed, admissible) ⇔ claimed ≤ admissible` — flags both
  strictly-too-strong **and incomparable** over-claims.

### T3 — Transfer falsifier (`mh/transfer_falsifier.py`)
One calculus, five unrelated domains; it independently flags **4/5** over-claims,
including a neuroscience over-claim shaped like the `ds003458` PLV null
(PLV `EXECUTED` but the frequency-band match only `ASSUMPTION` → admissible
`ASSUMPTION`, claimed `EXECUTED`) and an *incomparable* `argumentation` case.
All-`EXECUTED` software is correctly not flagged. This reproduces the **structure**
of the null at the inference-discipline layer; real EEG numbers are `BLOCKED`.

### T2 — Critic calibration (`mh/critic_calibration.py`, needs the semantic stack)
Four critic agents on **n = 16** gold items (95% Wilson CI on accuracy):

| agent | accuracy [95% CI] | precision | recall | Brier | ECE* |
|---|---|---|---|---|---|
| deterministic (keywords) | 0.688 [0.444, 0.858] | 1.00 | 0.375 | 0.313 | 0.313 |
| nli (contradiction prob)  | 0.938 [0.717, 0.989] | 1.00 | 0.875 | 0.061 | 0.097 |
| llm (qwen2.5:3b)          | 1.000 [0.806, 1.000] | 1.00 | 1.000 | 0.019 | 0.100 |
| red (adversarial)         | 0.500 [0.280, 0.720] | 0.50 | 1.000 | 0.403 | 0.363 |

Fleiss κ = 0.163. *ECE is **underpowered** at n=16 (5 bins) — indicative only.
The LLM's `P(stop)` is mapped from its discrete verdict, so its Brier reflects
that map, not the model's own probability.

### T4 — Verification scaling **curve** (`mh/verification_scaling.py`, needs the stack)
Routed verification quality vs verifier compute (relative units, **not** FLOPs).
This is a **4-point observation on one n=16 set, not a law**:

| n | stack | cost | routed acc | naive acc |
|---|---|---|---|---|
| 1 | det | 1 | 0.688 | 0.688 |
| 2 | +nli | 11 | 0.938 | 0.688 |
| 3 | +llm | 111 | **1.000** | 0.875 |
| 4 | +red | 211 | 1.000 | 0.938 |

Routed quality saturates at **n = 3**: the 4th verifier adds 0 routed gain at ~2×
compute. (Routing selects argmin-Brier **in-sample** — an unbiased estimate needs
a held-out split.)

### T5 — Breathing loop (`mh/breathing_loop.py`)
Self-tunes a fidelity threshold on a 12-item set: held-out F1 **0.0 → 0.8** (net
+0.8), **0 regressions**; the Goodhart bait (predict-all-faithful, which a naive
recall-optimiser would select) is **blocked by the tripwire** (1 event caught).

---

## 2. Falsification bounds (where it breaks)

- **Keyword critic**: precision 1.00 but recall **0.375** — blind to paraphrase
  and negation. Never use alone for safety.
- **Embedding cosine**: blind to **negation** by construction (negated thesis ~0.97).
- **Fidelity proxy (T5)**: two negation traps cap held-out F1 **below 1.0**.
- **Naive ensemble**: one miscalibrated rater (the adversary) drags the equal-vote
  mean **below the best single agent** (lift = **−0.0625**). Weighting does not fix
  it when one agent dominates; calibration **routing** (argmin Brier) does.
- **Local LLM critic**: qwen2.5:3b — a conservative gate, perfect on n=16 but a
  3B model with accuracy CI [0.806, 1.0]; do not extrapolate.

> Measured lesson: **average raters when they are peers; route when one dominates.**

---

## 3. Prior art & positioning

This is **not** a new uncertainty calculus. It is a small, runnable, **fail-closed
operational gate** built on an evidence-strength lattice. It is adjacent to, and
deliberately simpler than:

- **Dempster–Shafer** belief functions and **Jøsang's subjective logic** — full
  belief/uncertainty algebras; here we use only a finite strength **lattice** with
  a meet-as-conjunction rule, not mass assignment or fusion operators.
- **Data provenance / lineage lattices** — the `EXECUTED ▷ SOURCE ▷ …` order is a
  provenance-strength order with an admissibility gate.
- **Argumentation frameworks / paraconsistent logics** — the over-claim check is a
  minimal admissibility test, not a defeasible-reasoning engine.

The contribution is **operational**: a tested, fail-closed gate that refuses a
GREEN verdict whose weakest critical leg is too weak, plus measured calibration of
the critics that feed it.

---

## 4. Open metrics — latency, allocation, environment

- Pure tasks (T1/T3/T5): deterministic, sub-second, stdlib + sklearn, no GPU.
- Semantic stack (T2/T4): `sentence-transformers 5.5.1` (MiniLM embed + MiniLM-NLI)
  reusing a system `torch`; **no API key, no cost, local**.
- LLM agent: local **ollama / qwen2.5:3b** on an RTX 3050 (≈1.9 GB weights,
  ≈3.7 GB VRAM free). The 16-item T2 harness (× nli/llm/red) ran in **≈47 s** wall.
- Scaling-curve compute units are relative orders of magnitude (keyword 1,
  embedding 10, LLM 100) — **not** measured FLOPs.
- Recorded versions live in `reports/full_report.json → environment`.

---

## 5. Limitations (honest ceiling)

- **Small samples** (16/12/11) are the binding limit on every model-gated claim;
  these are **operating-point** measurements with wide CIs, not generalisation
  guarantees. Gold labels are single-author.
- T2/T4 require the semantic stack; without it they report `BLOCKED`, never faked.
- The neuroscience transfer is **structural**; real EEG is `BLOCKED` on data files.
- Commits are not GPG-signed (no signing key configured here).
- Full self-audit with every gap and its resolution: [`AUDIT.md`](AUDIT.md).

---

## Layout

```
mh/evidence_lattice.py     T1  bounded distributive lattice, 21 property tests
mh/transfer_falsifier.py   T3  pure
mh/breathing_loop.py       T5  pure (sklearn)
mh/critic_calibration.py   T2  semantic stack + ollama
mh/verification_scaling.py T4  semantic stack + ollama
mh/stats.py                Wilson CI + power flags
mh/run_all.py              aggregate -> artifacts/research_report.json
reports/                   committed pure_report.json + full_report.json
```

MIT licensed.
