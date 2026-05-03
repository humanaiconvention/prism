# SGT Recursive Grounding Benchmark — Pre-registration v1

Locked: 2026-05-03. Edits after this date go in an Addendum, never in this file.

## 1. Hypotheses (preserved from v9.2 §5)

| ID | Claim | Direction |
|----|-------|-----------|
| H1 | Recursive closure increases the probability of metric decoupling vs. corrected regimes. | Closed > Corrected |
| H2 | Increasing g_t / verification quality improves stability, possibly at different rates for accuracy and form. | Higher correction → better |
| H3 | PRISM `EarlyWarningAnalyzer` detects perplexity-first signature on closed regimes and not on adequately corrected regimes. | R1 fires; R4 clears |
| H4 | Δt sign is regime-dependent, not universally positive. | Distribution of Δt across regimes is non-degenerate |

## 2. Locked design

**Regimes** (5):
- `R1` synthetic-only replace
- `R1_accum` synthetic-only accumulate
- `R2` synthetic + frozen real-data anchor (50/50)
- `R3` synthetic + fresh real-data slice per generation (50/50)
- `R4` synthetic + teacher-corrected (50/50, teacher = next-larger Qwen2.5)

**Correction sweep** (separate, R4-style only): correction_frac ∈ {0, 25, 50, 75, 100}%.

**Models**: Qwen2.5-0.5B (primary), Qwen2.5-1.5B (scale check). Teacher for R4: Qwen2.5-1.5B for the 0.5B runs; Qwen2.5-7B (4-bit) for the 1.5B runs.

**Seeds**: 11, 23, 42. Locked. No reroll.

**Generations**: 5 per run.

**Eval anchors** (held out, never seen by recursive loop):
- Perplexity: WikiText-2 validation, first 500 lines.
- Accuracy: ARC-Easy test split, 200 items, exact-match on the answer letter.

**Synthetic generation**: 2000 samples/gen, 5-token real-prefix seed, top_k=50, max_new_tokens=128.

## 3. Locked thresholds

| Quantity | Threshold | Notes |
|---|---|---|
| t_perplexity | first gen with PPL ≥ 1.05 × baseline_PPL | baseline = gen 0 measurement |
| t_accuracy   | first gen with ACC ≤ 0.90 × baseline_ACC | baseline = gen 0 measurement |
| Δt           | t_perplexity − t_OOD | undefined → "censored" |
| seeds count for "passes" | ≥ 2 of 3 | majority rule |

If baseline ACC < 0.20, accuracy is treated as floor-censored and excluded from H1/H4 for that run (documented per-run, not after-the-fact).

## 4. Decision rules

| Hypothesis | Pass condition (per model scale) |
|---|---|
| H1 | (R1, R1_accum) median \|Δt\| ≥ 1 in ≥2/3 seeds AND (R2, R4) median \|Δt\| ≤ 1 in ≥2/3 seeds. **Both-censored R4 seeds count as \|Δt\|=0** (no decoupling = consistent with H1). |
| H2 | Correction-sweep ARC at 75–100% > ARC at 0–25% in ≥2/3 seeds. |
| H3 | EarlyWarningAnalyzer.detect signature fires on R1 in ≥2/3 seeds AND does not fire on R4 in ≥2/3 seeds. |
| H4 | At least one regime shows median Δt > 0 AND at least one shows median Δt < 0 (or censored), across the 5 regimes. |

A claim is "supported" if it passes at both 0.5B and 1.5B; "partial" if at one scale; "unsupported" otherwise.

## 5. Falsification rules (pre-committed)

- **Whole framework** is falsified if (a) all five regimes produce indistinguishable Δt distributions across seeds (H4 fails), or (b) increasing oracle-correction fraction does not improve ARC accuracy at either scale (H2 fails), or (c) `EarlyWarningAnalyzer` does not discriminate R1 from R4 (H3 fails).
- **Specific revised claim** (perplexity-first under R1 on small models) is falsified if, at 0.5B *and* 1.5B, R1 shows median Δt ≥ 0 in ≥2/3 seeds.

## 6. Analysis plan

- Per-run: log per-generation `{ppl, acc, gen, regime, seed, model}` to `runs/<model>/<regime>_<seed>.json`.
- Compute `t_perplexity`, `t_accuracy`, `Δt`, censoring status using `prism.eval.early_warning.EarlyWarningAnalyzer`.
- Bootstrap 95% CIs for `t_perplexity`, `t_accuracy`, `Δt` across 3 seeds (1000 resamples).
- Cross-regime contrast: paired comparison R1 vs R4 within seed; report the per-seed sign pattern.
- PRISM Lattice Viability Monitor: not used as a primary test; logged as exploratory diagnostic.

## 7. Stop conditions

- Compute exhaustion before all 30 runs complete: report partial results, do not extend the design.
- Discovery of a code bug after Weekend 1: re-run all affected configs from scratch with the same seeds; no cherry-picking.
- Surprise finding outside H1–H4: report in an "Exploratory observations" section only.

## 8. Reporting

The v9.3 paper reports H1–H4 outcomes verbatim against the table in §4 above. Anything else is exploratory and labeled as such.

---
*Author: B. Haslam. Pre-registered on Zenodo (concept DOI: 10.5281/zenodo.18091864) befor