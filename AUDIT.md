# Critical audit & remediation

A hostile self-review of `measured-honesty`. Every gap below was found by
attacking the repo's own claims; each is either **fixed** in code or **disclosed**
as a standing limitation. Nothing is hand-waved.

## Theoretical gaps

| # | Gap | Status |
|---|-----|--------|
| T-1 | Over-claim detection only caught *strictly-above* claims; an **incomparable** claim (e.g. `INFERENCE` claimed from a `SPECIFIED`-only premise) slipped through. | **FIXED** — `is_admissible(claimed, admissible) := claimed ≤ admissible`; over-claim ⇔ not admissible. New corpus case + test (`argumentation` domain). |
| T-2 | The lattice was asserted bounded but its **distributivity** was never proven. | **FIXED** — `is_distributive()` + property test; the carrier is distributive (no M3/N5). |
| T-3 | Only **conjunction** (meet) existed; a real evidence calculus also needs **disjunction** (alternative paths = join). | **FIXED** — `disjunction_strength = join` + property test (`conclusion ≥ every path`). |
| T-4 | No positioning against prior uncertainty calculi. | **DISCLOSED** — see README *Prior art*: this is an operational, fail-closed *gate*, not a new belief calculus; it relates to Dempster–Shafer, subjective logic and provenance lattices. |
| T-5 | `strength_rank` is a total index that ties incomparable elements. | **DISCLOSED** — documented as display-only, not the partial order. |

## Statistical / empirical gaps

| # | Gap | Status |
|---|-----|--------|
| S-1 | Metrics reported to 3–4 decimals on **n = 16/12/11** — false precision. | **FIXED/DISCLOSED** — per-agent accuracy now carries a 95% **Wilson CI** (`accuracy_ci95`); e.g. LLM 1.000 → **[0.806, 1.0]**, keyword 0.688 → **[0.444, 0.858]**. Caveat block added. |
| S-2 | **ECE** computed on 5 bins over 16 points — statistically meaningless. | **FIXED** — `ece_underpowered = True` flag; README marks ECE indicative-only. |
| S-3 | The LLM's `P(stop)` is **hand-mapped** from its discrete verdict, so its Brier reflects that map, not the model's own probability. | **DISCLOSED** — stated in `caveats` and README. |
| S-4 | Calibration **routing selects on the same gold set it is scored on** (in-sample, no held-out split). | **DISCLOSED** — `selection_rule` and caveats say "in-sample"; an unbiased estimate needs a held-out split. |
| S-5 | Gold labels are **author-authored** (single annotator); no inter-annotator agreement. | **PARTLY ADDRESSED** — added `mh/benchmark_snli.py`: the NLI critic on real SNLI (human gold, open) scores **0.904 acc / 0.904 macro-F1** on n=250. Honest scope: in-distribution (model trained on SNLI+MNLI), not zero-shot. The other gold sets remain author-authored. |

## Naming / honesty gaps (the project's own thesis)

| # | Gap | Status |
|---|-----|--------|
| H-1 | T4 called a **"scaling law"** on a 4-point curve — exactly the kind of over-claim this repo forbids. | **FIXED** — renamed to an **observation / curve**; `caveat` says "4 points, not a law". |
| H-2 | Compute "costs" were relative units presented like measurements. | **DISCLOSED** — labelled "relative orders of magnitude, not measured FLOPs". |
| H-3 | T3 "transfer" is structural-by-construction, not empirical. | **DISCLOSED** — README: reproduces the *structure* of the `ds003458` null at the inference-discipline layer; real EEG numbers are `BLOCKED`. |

## Reproducibility / engineering gaps

| # | Gap | Status |
|---|-----|--------|
| R-1 | `research_report.json` was **gitignored** — README numbers were unverifiable from the repo. | **FIXED** — `reports/pure_report.json` (model-free) and `reports/full_report.json` (with versions) are committed. |
| R-2 | No recorded **library/model versions** — numbers wouldn't reproduce. | **FIXED** — `environment` block in the report (sklearn 1.8.0, sentence-transformers 5.5.1, qwen2.5:3b, MiniLM models). |
| R-3 | No pinned dependencies. | **FIXED** — `requirements.txt` / `requirements-semantic.txt`. |
| R-4 | No CI. | **FIXED** — `.github/workflows/ci.yml` runs ruff + mypy + pure tests on every push. |
| R-5 | LLM determinism across ollama/GPU versions unverified. | **DISCLOSED** — temp 0 is mostly-deterministic; not guaranteed across versions. |

## Still open (honest)

- Small samples are the binding limit on every model-gated claim (S-1/S-5).
- No real EEG transfer (H-3) — `BLOCKED` on data files.
- Commits are not GPG-signed (no signing key configured).
- `_fleiss_kappa` returns 0.0 on a balanced 2-rater toy; consistent with chance
  agreement but not cross-validated against a reference implementation.
