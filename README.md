# measured-honesty

**Five falsification studies of an instrument's own limits.**
Every number below is produced by an executed measurement in this repo. There are
no marketing figures and no hypotheticals — claims that cannot be measured are
marked `BLOCKED` or `UNKNOWN`, never rounded up.

```
make verify        # lint + type + pure tests + report   (T1/T3/T5, no GPU)
make report        # writes artifacts/research_report.json
```

Reproduce any line in this README from `artifacts/research_report.json`.

---

## Why this exists

Most tools publish what they can do. This one publishes **where it breaks** and
**how well-calibrated it is about its own judgements** — measured, not asserted.
The thesis: an instrument earns trust by mapping its own failure manifold.

### First-principle framing (isomorphisms, not claims)

These are analogies that motivated the design, not empirical assertions:

- **Active Inference** — a verdict is only as strong as the *precision* of its
  weakest critical premise. T1 makes that a literal algebra (`conjunction = meet`).
- **Attention Schema Theory** — a system that models its own attention is more
  controllable. T2 builds a *schema of the critic itself*: its calibration
  (Brier/ECE) is the system's model of how much to trust its own attention.
- **Goodhart / free-energy** — optimising a proxy of success is not success. T5
  installs a tripwire that refuses a proxy win that collapses the prediction.

---

## 1. Boundary capabilities (measured)

### T1 — Evidence lattice (`mh/evidence_lattice.py`)
Evidence classes form a **bounded lattice** under epistemic strength
(`UNKNOWN < ASSUMPTION < {SPECIFIED, INFERENCE} < SOURCE < EXECUTED`). Two laws are
proven as algebra by **16 Hypothesis property tests**:

- `admits_green(e)` is true **iff** `e ≥ SOURCE` — a critical claim cannot be GREEN
  on a weak premise.
- `conjunction_strength = meet` — a conclusion is no stronger than its weakest
  critical leg. `conj(EXECUTED, ASSUMPTION) = ASSUMPTION` → not GREEN.

### T3 — Transfer falsifier (`mh/transfer_falsifier.py`)
One calculus, four unrelated domains. It independently flags **3/4** over-claims,
including a neuroscience over-claim shaped like the `ds003458` PLV null
(PLV `EXECUTED` but the frequency-band match only `ASSUMPTION` → admissible
strength `ASSUMPTION`, claimed `EXECUTED`). The software case (all `EXECUTED`) is
correctly *not* flagged.

### T2 — Critic calibration (`mh/critic_calibration.py`, needs the semantic stack)
Four independent critic agents on a 16-item gold set:

| agent | acc | precision | recall | Brier | ECE |
|---|---|---|---|---|---|
| deterministic (keywords) | 0.688 | 1.00 | 0.375 | 0.313 | 0.313 |
| nli (contradiction prob)  | 0.938 | 1.00 | 0.875 | 0.061 | 0.097 |
| llm (qwen2.5:3b)          | 1.000 | 1.00 | 1.000 | 0.019 | 0.100 |
| red (adversarial)         | 0.500 | 0.50 | 1.000 | 0.403 | 0.363 |

Fleiss κ = 0.163 (low agreement, driven by the adversary).

### T4 — Verification scaling law (`mh/verification_scaling.py`, needs the stack)
Routed verification quality vs verifier compute (relative units):

| n | stack | cost | routed acc | naive acc |
|---|---|---|---|---|
| 1 | det | 1 | 0.688 | 0.688 |
| 2 | +nli | 11 | 0.938 | 0.688 |
| 3 | +llm | 111 | **1.000** | 0.875 |
| 4 | +red | 211 | 1.000 | 0.938 |

Routed quality **saturates at n = 3**: the 4th verifier adds 0 routed gain at ~2×
compute.

### T5 — Breathing loop (`mh/breathing_loop.py`)
Self-tunes a fidelity threshold: held-out F1 **0.0 → 0.8** (net +0.8), **0
regressions**, and the Goodhart bait (predict-all-faithful, which a naive
recall-optimiser would select) is **blocked by the tripwire** (1 event caught).

---

## 2. Falsification bounds (the exact conditions where it breaks)

- **Keyword critic**: precision 1.00 but recall **0.375** — blind to paraphrased
  manipulation and to negation. Do not use alone for safety.
- **Embedding cosine**: blind to **negation** by construction (a negated thesis
  scores ~0.97 vs the thesis). Faithfulness needs the NLI head, not cosine.
- **Fidelity proxy (T5)**: two negation traps cap held-out F1 strictly **below
  1.0** — a structural metric cannot fix a meaning-flip.
- **Naive ensemble**: a single miscalibrated rater (the adversary) drags the
  equal-vote mean **below the best single agent** (lift = **−0.0625**).
  Calibration *weighting* does not fix it when one agent dominates;
  calibration *routing* (argmin Brier) does (lift 0.0).
- **Local LLM critic**: qwen2.5:3b is a **conservative** gate, perfect on this
  16-item set but a 3B model, not a frontier judge — do not extrapolate.

> Lesson, measured: **average raters when they are peers; route when one dominates.**

---

## 3. Open metrics — latency & allocation

- Pure tasks (T1/T3/T5): deterministic, sub-second, no GPU, stdlib + sklearn.
- Semantic stack (T2/T4): `sentence-transformers` (MiniLM embed + MiniLM-NLI) reusing
  system `torch`; **no API key, no cost, fully local**.
- LLM agent: local **ollama / qwen2.5:3b** on an RTX 3050 (≈1.9 GB weights,
  ≈3.7 GB VRAM free). The 16-item T2 harness (× nli/llm/red) completed in
  **≈47 s** wall on that GPU.
- Scaling-curve compute units are relative orders of magnitude:
  keyword = 1, embedding = 10, LLM = 100.

---

## Honest ceiling (`BLOCKED`, not hidden)

- T2/T4 require the semantic stack; without it they report `BLOCKED`, never faked.
- Gold/benchmark sets are small (16/12/11 items) — these are **operating-point
  measurements**, not generalisation guarantees.
- The neuroscience transfer reproduces the *structure* of the `ds003458` null at
  the inference-discipline layer; wiring the real EEG numbers is future work,
  `BLOCKED` on the data files.
- Commits in this repo are not GPG-signed (no signing key configured here).

---

## Layout

```
mh/evidence_lattice.py     T1  pure, 16 Hypothesis property tests
mh/transfer_falsifier.py   T3  pure
mh/breathing_loop.py       T5  pure (sklearn)
mh/critic_calibration.py   T2  semantic stack + ollama
mh/verification_scaling.py T4  semantic stack + ollama
mh/run_all.py              aggregate -> artifacts/research_report.json
```

MIT licensed. Built by applying a fail-closed evidence discipline to its own
measurements.
