# SGT v11 — Deviations from `sgt-prereg-v1`

The preregistration was locked on 2026-05-03 at git tag `sgt-prereg-v1`, commit
`6092c525797452ea811d9f834a7757f6445abc64` (tag object
`606272dbc477094475d265036f3f76468d37a952`) in the local monorepo.

Patches landed on the runner between the locked tag and the v11 sweep are
recorded below. Per v10.6's deviations protocol, these are runner / environment
adaptations only. None of them changes:

- Regime definitions (R1, R1_accum, R2, R3, R4)
- Threshold values (PPL ≥ 1.05 × baseline; ACC ≤ 0.90 × baseline; ACC floor 0.20)
- Decision rules (≥ 2 of 3 seeds; H1/H2/H3/H4 wording)
- Primary metrics (WikiText-2 validation perplexity, ARC-Easy test accuracy)
- Sample design (5 regimes × 3 seeds × 2 model scales × 5 generations)

All commit hashes are from the monorepo `D:/humanai-convention`. The
corresponding "(sync from monorepo)" commits in the public `D:/prism` mirror
carry the same content.

---

## Patches (in commit order, all 2026-05-03)

### 1. Two setup-time bugfixes
- **Commit**: `6af6264bbd11e2c76bd83acab3169a500ff772a2`
- **Change**: Fixed two setup-time bugs in `sgt_runner.py` (CLI parsing /
  initialization).
- **Why**: Code did not start under the locked configuration without these
  fixes; the design itself was unaffected.
- **Affects decision rules?** No.

### 2. gradient_checkpointing + Colab notebook fixes
- **Commit**: `278018810b5dbc1fb411748c5205497f7781afbf`
- **Change**: Enabled `gradient_checkpointing` during fine_tune; minor Colab
  notebook environment fixes.
- **Why**: Memory headroom for 0.5B/1.5B fine-tune steps on T4 16 GB. Trades a
  ~20–30% step-time increase for the headroom needed to actually complete a
  generation without OOM.
- **Affects decision rules?** No. The training procedure is unchanged in
  substance; gradient_checkpointing is a memory/compute trade-off, not a
  modelling change.

### 3. Restore `model.config.use_cache=True` after fine_tune
- **Commit**: `ebe67cc2a5dafe143824483654944f4239878b5e`
- **Change**: After fine-tuning (which sets `use_cache=False` for
  gradient_checkpointing), restore `use_cache=True` before the next generation
  step.
- **Why**: Generation speed during synthetic-corpus production. Pure runtime
  property; output sequences are identical given the same RNG seed.
- **Affects decision rules?** No.

### 4. Batched synthetic generation (~20–30× speedup)
- **Commit**: `580090b9689d5caa2e71f481f4f924b4942b42db`
- **Change**: `make_synthetic` now batches prompts instead of generating one at
  a time.
- **Why**: ~20–30× wall-clock speedup on T4; without it the 30-run primary
  sweep is impractical.
- **Affects decision rules?** No, **with one important caveat documented in
  v10.6 and below**.
- **Caveat — RNG ordering between sequential and batched generation**: Pilot
  work observed that sequential vs batched generation can diverge per-token at
  the same seed because they consume random numbers in different orders. The
  v11 runner now uses batched generation **exclusively**. Sequential generation
  is not used as an alternative path in any v11 regime. This pin removes the
  procedure as an unreported analysis degree of freedom; it does not change the
  preregistered design, which never specified sequential generation. The
  companion methodology paper will report this observation in full.

### 5. Default `CUDA_VISIBLE_DEVICES=0`
- **Commit**: `7cbf5f318a35944c3fc9c17a3eaec2d5206d2b9e`
- **Change**: Default `CUDA_VISIBLE_DEVICES=0` to avoid Hugging Face Trainer's
  automatic `DataParallel` wrapping on multi-GPU machines (e.g. Kaggle T4×2).
- **Why**: Auto-`DataParallel` changes effective batch size and gradient
  reduction behaviour relative to the locked single-GPU configuration. Pinning
  to a single device preserves the configuration that was preregistered.
- **Affects decision rules?** No. This *prevents* an environmental drift that
  would have invalidated the preregistered configuration.

### 6. Split `batch_size` into `per_device_bs` + `grad_accum`
- **Commit**: `957cfe4b5c78a6d62e157293540d3320475d14f5`
- **Change**: Effective batch size is now `per_device_bs × grad_accum`,
  defaulting to `4 × 2 = 8` to fit T4 16 GB.
- **Why**: 0.5B with the original `batch_size=8` did not fit at T4 16 GB
  alongside the activations needed for fine-tune. The effective batch size of
  8 is preserved; only its decomposition changes.
- **Affects decision rules?** No. Effective batch size is unchanged.

---

## R4 teacher model pin (v11 sweep decision)

The preregistration specifies R4 as "50/50 synthetic + teacher-corrected" but
does not pin a specific teacher model — only that the teacher provides
deterministic continuations of the same prefixes. This is now pinned for the
v11 sweep.

- **0.5B sweep teacher**: `Qwen/Qwen2.5-7B-Instruct`, loaded with 8-bit
  quantization via `bitsandbytes` (`load_in_8bit=True`), `device_map="auto"`.
- **1.5B sweep teacher**: same model. The trainee/teacher ratio drops from
  ~14× (7B vs 0.5B) to ~4.7× (7B vs 1.5B); the teacher remains meaningfully
  more capable than the trainee, which is what R4 requires to be a
  high-correction regime distinct from R3.
- **Teacher decoding policy**: `do_sample=False, max_new_tokens=128`
  (deterministic greedy). Hard-coded in `sgt_runner.py:teacher_relabel`.
- **Correction fraction for R4**: `correction_frac=0.5` by default in the
  runner; the regime spec passes this through. R4 therefore relabels 50% of
  the synthetic items per generation. The `Rn_<frac>` regime variant exposes
  this as a sweep parameter for sensitivity analysis (out of scope for the
  primary 30-run sweep).
- **Smoke-test plan**: validate the R4 code path with `Qwen2.5-1.5B-Instruct`
  as a cheaper smoke teacher first, then switch to `Qwen2.5-7B-Instruct` for
  the production sweep once memory fit is confirmed.

**Memory fit on T4 16 GB (0.5B trainee + 7B teacher 8-bit)**:
- Trainee fp32 (~2 GB) + teacher 8-bit (~7 GB) during `teacher_relabel`: ~9 GB. Fits.
- Same + Trainer activations during `fine_tune` (~5–7 GB): ~14–16 GB. Tight.
- Fallback if OOM: switch teacher to 4-bit quantization (`load_in_4bit=True`
  via `BitsAndBytesConfig`). Recorded as patch #7 if applied.

**Does pinning the teacher affect decision rules?** No. The preregistered
text says "teacher-corrected" and pins the *role* of the teacher, not the
specific weights. The model identity is published here so that v11 is fully
reproducible from the public PRISM repo + this DEVIATIONS file.

---

## Kaggle environment patch (2026-05-21)

### Patch K1: drop the `torch==2.5.1` reinstall in Kaggle kernels
- **Where**: Kaggle notebook builder scripts at
  `D:/kaggle/scripts/build_sgt_v11_smoke_nb.py` and
  `D:/kaggle/scripts/build_sgt_v11_0p5b_nb.py`.
- **Change**: Removed the line
  `pip install -q torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121`.
  Kaggle's T4/T4x2 base image already ships a compatible `torch` + `torchvision`
  pair (verified torch 2.5.1+cu121 already present at session start). The
  prior reinstall pinned torch without a matching torchvision and broke
  `torchvision::nms` C++ op registration, causing `transformers.modeling_utils`
  to fail to import in ~4 s.
- **Symptom in v1 of the smoke kernel** (`benhaslam/sgt-v11-stage-0-smoke`,
  2026-05-21 23:32 UTC): both smoke runs crashed at import with
  `RuntimeError: operator torchvision::nms does not exist`. Total wall-clock
  ~3.5 min; no runs/ output produced.
- **Affects decision rules?** No. This is an environmental adaptation
  specific to Kaggle's package management. The `sgt_runner.py` source is
  unchanged.
- **Also added**: a fail-fast import sanity check at the end of Cell 1
  (touches `torch.ops.torchvision.nms` and imports `transformers.Trainer`)
  so the same failure mode surfaces in seconds rather than after the runner
  starts.

### Patch K2: bump `bitsandbytes` pin from 0.44.1
- **Where**: Same builder scripts.
- **Change**: `bitsandbytes==0.44.1` → `bitsandbytes>=0.45.0` (pip picks the
  highest compatible). 0.44.1 imports `triton.ops.matmul_perf_model`, which
  no longer exists in the triton version Kaggle ships in 2026.
- **Symptom in v2** (2026-05-21 23:50 UTC): `ModuleNotFoundError: No module
  named 'triton.ops'` at the import sanity check, kernel ERROR at 59 s.
- **Affects decision rules?** No.
- **Also added**: `from bitsandbytes.nn import Linear8bitLt` to the sanity
  check, so triton-backed module load failures surface at Cell 1.

### Patch K3: pin GPU to T4 via `machine_shape`
- **Where**: `kernel-metadata.json` for both smoke and 0p5b sweep kernels.
- **Change**: Added `"machine_shape": "NvidiaTeslaT4"`. Without this, Kaggle
  free-tier accelerator can assign Tesla P100 (sm_60), which the current
  torch wheels do not support — the wheels target sm_70+ (T4 is sm_75).
- **Symptom in v3** (2026-05-22 00:00 UTC): Got assigned a P100; both smokes
  ran imports OK but failed at first CUDA op with
  `torch.AcceleratorError: CUDA error: no kernel image is available for
  execution on the device`. Smoke A failed at 122 s, smoke B at 163 s
  (after partial teacher download).
- **Affects decision rules?** No. The preregistered design specifies model
  scales (Qwen2.5-0.5B, 1.5B) but not the host GPU; pinning to T4 is an
  environmental choice. We will use the same T4 pin for the full sweep so
  hardware is consistent across runs.

### Patch K4: lazy-load R4 teacher per generation
- **Where**: `experiments/sgt/sgt_runner.py` (run function, teacher loading +
  per-generation loop). Commit `86d6e56` on PRISM main (2026-05-21).
- **Change**: The teacher was previously loaded once at the top of `run()`
  and kept resident across all generations. Now the teacher is loaded
  just-in-time inside the per-generation loop, immediately before
  `teacher_relabel`, and `del`'d + `gc.collect()` + `torch.cuda.empty_cache()`
  immediately after. The trainee's `fine_tune` step thus runs with no teacher
  resident.
- **Why (v4 evidence)**: The v4 smoke (0.5B trainee + 1.5B-Instruct teacher
  in 8-bit) ran for 13,797 s (3.83 hours) before OOMing during gen-5
  `fine_tune` backward pass. Final memory state at OOM: T4 had 14.56 GiB total,
  29.81 MiB free; PyTorch had 13.14 GiB allocated, 1.25 GiB reserved.
  Co-residency of even the smaller smoke teacher with the trainee's
  activations + grads is not viable on T4. With the production
  `Qwen2.5-7B-Instruct` teacher this would have been worse.
- **Trade-off**: Loading the 8-bit teacher each generation costs ~10-30 s
  for the 1.5B-Instruct smoke teacher, ~30-60 s for the 7B-Instruct production
  teacher. Over 5 generations × 3 seeds × 1 regime (R4) that is at worst
  ~15 minutes per scale. Per-gen `fine_tune` is ~15 minutes on T4, so the
  load overhead is small relative to the regime's total wall time.
- **Affects decision rules?** No. The teacher_relabel function call site,
  arguments, and decoding policy are unchanged. The only externally
  observable difference is wall-clock time. Per-gen RNG state is unaffected
  because the teacher's `do_sample=False` greedy decode is deterministic
  given the same prefix.

**Sanity statement for the v11 paper**: "R4's teacher model is loaded
inside the per-generation loop and released before the trainee's fine_tune
step to make teacher + trainee co-residency unnecessary on 16 GB-class
accelerators. This is a memory-management decision; the per-generation
teacher_relabel logic, deterministic greedy decoding, and correction
fraction are unchanged."

### Patch K5: batch `teacher_relabel`
- **Where**: `experiments/sgt/sgt_runner.py:teacher_relabel`. PRISM main
  commit `8e5f4a2`.
- **Change**: The pre-v11 `teacher_relabel` looped one sample at a time
  through `teacher_model.generate`. Now it batches with left-padding
  (`batch_size=8` default), matching the `make_synthetic` pattern.
- **Why (v5 evidence)**: With the production teacher `Qwen2.5-7B-Instruct`
  in 8-bit on T4, the single-sample loop took ~230 minutes per generation
  for 1000 relabel items. v5 was killed by Kaggle's session cap at 3.91
  hours after completing only gen 1. Batched generation reduces per-gen
  wall-clock by roughly 10x for this teacher/trainee/batch combination.
- **Output equivalence**: greedy decoding (`do_sample=False`) is invariant
  to batch shape modulo left-padding; the same prefix produces the same
  continuation regardless of whether it is in a batch of 1 or a batch of 8.
- **Affects decision rules?** No.

### Patch K6: empty CUDA cache after `fine_tune` in R4 loop
- **Where**: `experiments/sgt/sgt_runner.py` per-generation loop, after
  the `fine_tune` call, only when `oracle == "teacher_filter"`. PRISM
  main commit `8e5f4a2`.
- **Change**: `gc.collect()` + `torch.cuda.empty_cache()` after
  `fine_tune` and before the next iteration's `make_synthetic` call.
- **Why (v5 evidence)**: After fine_tune, the optimizer state and gradient
  buffers attached to the trainee continue to occupy ~5-6 GB of T4 VRAM
  through the next iteration's `make_synthetic` and on into the next
  teacher load. Without K6, accelerate's `device_map="auto"` offloads
  teacher weights to CPU on the gen-N+1 load. Once a teacher layer is
  on CPU, every batched generate() call shuffles weights across the
  PCIe bus, slowing `teacher_relabel` another ~10x even after K5.
  v5 log showed the CPU-offload warning at exactly the gen-2 teacher load.
- **Affects decision rules?** No. Pure memory hygiene.

**Kaggle session-cap observation (not a patch)**: v5 was killed at 3.91
hours with the message "exceeded the max allowed execution duration."
For free-tier T4 commit-mode notebooks Kaggle currently enforces a
duration shorter than the historical 12-hour cap. The v11 sweep is
therefore split into 3 sub-sweeps each fitting in ~2.5-3 hours of
wall-clock with the K5+K6 patches applied. Documented in
`D:/humanai-convention/experiments/improvement/sgt_v11/v11_plan.md`.

---

## Summary statement for the v11 paper

> "Six patches were applied to the runner between the preregistration lock
> (`sgt-prereg-v1`, 2026-05-03) and the v11 sweep, all of them addressing
> environmental compatibility (memory headroom on T4, single-GPU pinning,
> batched generation speedup, and setup-time bugfixes). The locked design,
> primary metrics, threshold values, and decision rules are unchanged. The
> batched-generation pin removes a procedural degree of freedom that pilot
> work showed could interact with RNG ordering; it does not alter any
> preregistered claim. Full patch-by-patch documentation is in
> `experiments/sgt/runs/DEVIATIONS.md`."

---

## Process going forward

Any further patches to `sgt_runner.py`, `sgt_analysis.py`, `plot_bands.py`, or
`format_outcome_table.py` between this file's creation date (2026-05-21) and
the v11 paper deposit must:

1. Land on the PRISM main branch with a `runner-patch:` or `analysis-patch:`
   commit prefix.
2. Be appended to this file under a new dated section with: commit hash, what
   changed, why, and whether any decision rule is affected.
3. Be declared in v11's "Deviations from the preregistration" subsection of
   §6.

Forbidden post-lock changes:
- Regime definitions (R1, R1_accum, R2, R3, R4)
- Threshold values (PPL +5%, ACC −10%, floor 0.20, ≥ 2/3 seeds)
- Primary metrics (WikiText-2 perplexity, ARC-Easy accuracy)
- Sample sizes (3 seeds, 2 scales, 5 generations)

If a forbidden change becomes necessary (e.g. an unrecoverable runner bug
makes a regime non-executable), the correct response is to:

1. Declare it in this file as a *blocked* deviation
2. Truncate the design to the executable subset
3. Report the truncation prominently in the v11 paper
4. Treat the un-executable regime as preregistered-but-not-run

Do not silently change a threshold or metric to make the data look cleaner.

---

## LL-ACC secondary metric (pre-registered addendum, 2026-05-29)

### Patch L1: additive LL-ACC recording in sgt_runner.py
- **Where**: `experiments/sgt/sgt_runner.py`, commit `bf8f82b` on PRISM main.
- **Change**: Added `eval_arc_llacc` (length-normalized + raw log-likelihood ARC
  accuracy) and recorded two new per-generation fields,
  `grounded_arc_acc_llnorm` and `grounded_arc_acc_llraw`, at the gen-0 baseline
  and every generation. The preregistered substring grader (`eval_arc`,
  `grounded_arc_accuracy`) is unchanged.
- **Why**: The substring grader floor-censors every 0.5B run (baseline 0.025 <
  0.20), making H1/H4 untestable. LL-ACC baseline ~0.545 un-floors ACC so the
  prereg's §4 H1/H3/H4 rules become evaluable. Decision rule locked pre-data in
  `experiments/sgt/preregistration_addendum_ll_acc.md` (`d47c2b8`).
- **Verified**: the runner's own `eval_arc_llacc` reproduces the C2 baseline
  (acc_norm 0.545 / acc_raw 0.590, fresh Qwen2.5-0.5B, ARC-Easy test[:200]) by
  direct GPU run.
- **Affects decision rules?** No locked rule changes. acc_norm is a declared
  secondary metric; substring remains the preregistered primary on record.
