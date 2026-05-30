# Pre-registration addendum: LL-ACC re-analysis of the v11 0.5B sweep

**Status: PRE-DATA.** Written 2026-05-28, before any LL-ACC re-run exists. The
purpose of this document is to lock the decision rule for the log-likelihood
accuracy (LL-ACC) re-analysis *before* observing its results, so that the
grader change cannot be construed as post-hoc grader-shopping to rescue H1/H4.

This addendum does NOT alter the original preregistration (`sgt-prereg-v1`,
commit `6092c525...`, locked 2026-05-03). The substring grader remains the
preregistered primary metric and its result (all runs floor-censored, H1/H4
not testable at 0.5B) stands as the preregistered outcome. LL-ACC is a
**declared secondary analysis** with its own locked rule below.

## Motivation (established, not assumed)

The C2 probe (`probe_ll_acc.py`, `ll_acc_probe.json`) established, on a fresh
Qwen2.5-0.5B baseline against the identical 200-item ARC-Easy slice:

- Substring grader (preregistered): 0.025 ACC — replicates the runner exactly,
  and is largely spurious (several "correct" items are first-letter
  coincidences, e.g. "Antarctica…" scored correct for key "A").
- LL-ACC (raw): 0.590; LL-ACC (length-normalized): 0.545.

So the 0.20 ACC floor that censored every run is a property of the substring
grader, not of the model's ARC competence. This motivates re-scoring, but the
*motivation* does not get to choose the *decision rule* after the fact — that is
what this addendum locks.

## Locked LL-ACC re-analysis design

1. **Metric.** Length-normalized log-likelihood accuracy (acc_norm) is the
   primary LL metric, computed exactly as in `probe_ll_acc.py`: for each ARC
   item, score `log P(choice_text | prompt)` for each candidate, length-normalize
   by token count, argmax, compare to answerKey. Raw-LL acc is reported as a
   secondary cross-check. Rationale for choosing acc_norm as primary: it is the
   `lm-eval-harness` default for ARC and is less biased toward short answers;
   locked here before seeing per-generation results.

2. **Prompt format.** The single format `"Question: {q}\nAnswer: {a}"` (the
   format used in C2 and the runner), locked. Format-robustness of the *baseline*
   is checked separately (`probe_ll_acc_formats.py`) but the re-run uses this one
   fixed format for all regimes/seeds/generations.

3. **Threshold.** The preregistered ACC degradation threshold is unchanged:
   t_accuracy = first generation at which acc_norm <= 0.90 × baseline acc_norm
   (i.e. a 10% relative drop), with the same 0.20 absolute floor exclusion. With
   baseline acc_norm ≈ 0.55, the 0.20 floor will not bind, so runs become
   eligible for H1/H4.

4. **Δt and hypotheses — the LOCKED §4 rules are evaluated verbatim.** What the
   LL-ACC re-run changes is *not* the decision rules: it changes only whether
   `t_accuracy` is computable. Under the preregistered substring grader, baseline
   ACC = 0.025 < 0.20 floor, so every run is floor-excluded and `t_accuracy`
   (hence Δt and |Δt|) is undefined — H1/H3/H4 cannot be evaluated. With
   acc_norm baseline ≈ 0.55 the floor does not bind, `t_accuracy` becomes
   defined, and we then evaluate the original preregistration's §4 decision
   rules **exactly as written**, substituting acc_norm-derived `t_accuracy`:

   - **H1 (locked, verbatim):** (R1, R1_accum) median |Δt| ≥ 1 in ≥2/3 seeds
     AND (R2, R4) median |Δt| ≤ 1 in ≥2/3 seeds. *(Both-censored R4 seeds count
     as |Δt| = 0, per prereg §4.)* Note the locked rule references only R1,
     R1_accum, R2, R4 — **R3 is not in the H1 rule**; we do not add it.
   - **H3 (locked, verbatim):** `EarlyWarningAnalyzer` fires on R1 in ≥2/3 seeds
     AND clears on R4 in ≥2/3 seeds.
   - **H4 (locked, verbatim):** ≥1 regime shows median Δt > 0 AND ≥1 shows
     median Δt < 0 (or censored), across the 5 regimes.

   `t_OOD ≡ t_accuracy` here (ARC-Easy is the grounded/OOD anchor; the prereg
   uses the two names interchangeably). No threshold, seed, regime, or rule
   wording is changed; only the ACC grader feeding `t_accuracy` differs.

5. **Falsification = the prereg's own §5, unchanged.** Whole-framework
   falsification remains the locked prereg §5 (H4 degenerate, OR H2 sweep shows
   no correction benefit, OR H3 fails to discriminate R1 from R4). The specific
   revised claim (perplexity-first under R1 on small models) is falsified if R1
   shows median Δt ≥ 0 in ≥2/3 seeds. We add **no new success criterion**.
   Explicitly: LL-ACC merely clearing the floor makes H1/H3/H4 *testable*, not
   *passed*; and a result where ACC degrades equally across regimes, or never
   degrades, is a publishable negative reported as such.

6. **Reporting both.** The paper reports substring-ACC (preregistered,
   floor-censored, H1/H4 not-testable) AND acc_norm LL-ACC (this addendum) side
   by side, with the grader discrepancy disclosed as a methodological finding.
   Neither is hidden.

7. **Confound disclosure (carried into the re-analysis).** The PPL axis is
   partially confounded — R2/R3 are fed real WikiText and evaluated on WikiText
   PPL. The ARC accuracy axis is NOT confounded (no regime trains on ARC). The
   re-analysis therefore treats the ACC-axis H1/H3/H4 result as the cleaner test
   of the framework, and will say so.

## §6 Secondary / exploratory analyses (NEW in this addendum — NOT locked decision rules)

These are added by this addendum and are explicitly **not** substitutes for the
prereg's §4 rules. They are reported as exploratory:

- **Correction-rank trend.** Kendall τ-b between the theory-specified 3-level
  correction rank ({R1,R1_accum}=0, {R2}=1, {R3,R4}=2) and a degradation
  magnitude (gen-5 PPL, and — once defined — gen-5 ACC drop). This is the
  statistic used descriptively on the 0.5B PPL data (τ = −0.82); applying it to
  the ACC axis under LL-ACC is exploratory, not a pass/fail gate.
- Per-generation trajectory shapes, training-loss mechanism, and effect sizes
  as in the G1 analysis.

## Deviations carried from the executed sweep (disclosed, not introduced here)

The LL-ACC re-run will re-use the existing sweep's configuration for
within-study comparability, which means it inherits two deviations from the
locked prereg that must be disclosed wherever results are interpreted:

- **R4 teacher.** Prereg §2 specifies the 0.5B-run teacher as **Qwen2.5-1.5B**
  (base, next-larger model). The executed R4 runs used **Qwen2.5-7B-Instruct
  (8-bit)** — verified in all three R4 JSONs; logged in DEVIATIONS K4. The
  scientific consequence: the executed R4 is "correction by a much stronger
  instruction-tuned model," not the preregistered "next-larger base model." Any
  R4-based reading (e.g. "R4 ≈ R3") is about a *stronger-than-preregistered*
  teacher and is labelled as such. The LL-ACC re-run keeps the 7B-Instruct
  teacher so its results are comparable to the existing sweep.
- **Grader implementation.** Prereg §2 says "exact-match on the answer letter";
  the runner implements *first-character* match (`ans[:1].upper() ==
  gold.upper()`), which is looser and produces spurious matches (e.g.
  "Antarctica…" → "A"). A true exact-letter grader would likely score ≤ 0.025,
  so the substring floor problem is if anything understated. The paper describes
  the grader precisely as implemented.

## H2 status (untested)

H2 (correction-fraction sweep, prereg §1/§2, correction_frac ∈ {0,25,50,75,100}%)
was **not run** — neither in the main sweep nor in this re-analysis. Therefore an
H1/H4 pass under LL-ACC is at most "partial" support (prereg §4: full support
requires passing at both scales and across the named hypotheses). H2 remains
untestable until the correction-fraction sweep is executed; this addendum does
not change that.

## Compute plan (no new design knobs)

Add LL-ACC scoring alongside the existing substring scoring in `sgt_runner.py`
(additive — the preregistered substring metric is still recorded). Re-run the
same 5 regimes × 3 seeds on Qwen2.5-0.5B next Kaggle quota cycle. Per-run
overhead ≈ +15-20% (one extra forward pass per candidate per item per gen).
No change to training, regimes, thresholds, or seeds.

## Honesty note

If the LL-ACC re-run does not support H1/H4 (e.g. ARC is recursion-robust and
never degrades, or degrades equally across regimes), that is a publishable
negative result and will be reported as such. The point of this addendum is
narrow and firm: it does **not** alter any locked decision rule — it only makes
`t_accuracy` computable so the prereg's existing §4/§5 rules can finally be
evaluated at 0.5B — and it discloses the two deviations (R4 teacher, grader
implementation) already present in the executed sweep. The success/failure
criteria are the preregistration's own, fixed before the re-run data exist.

---

## Supporting artifacts

The primary LL metric (`acc_norm`) is defined precisely and reproducibly by the
co-committed scorer `ll_acc_eval.py` (its `--selftest` reproduces the baseline
acc_norm ≈ 0.545 / acc_raw ≈ 0.590 on a fresh Qwen2.5-0.5B over ARC-Easy
test[:200]). The baseline probe and prompt-format robustness check that motivated
this addendum live in the project's experiment workspace and are reported in the
accompanying analysis; they are *motivation*, not part of the locked decision
rule. This addendum sits beside `preregistration.md` (locked 2026-05-03,
`sgt-prereg-v1`); it adds no rule and changes no threshold.
