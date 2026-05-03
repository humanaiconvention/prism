# SGT v9.3 Run Plan — fits Colab Pro (45 hrs) + Kaggle (~120 hrs/mo)

## Compute envelope (locked)

| Platform | Per month | Used here | Headroom |
|---|---|---|---|
| Colab Pro (L4 / A100) | ~45 hrs | ~25 hrs | 20 hrs reroll budget |
| Kaggle T4×2 / P100 | ~120 hrs | ~50 hrs | 70 hrs reroll budget |
| **Total** | **~165 hrs** | **~75 hrs** | comfortably 2× |

## Per-run estimates (measure after smoke test, update this table)

| Config | Wall time / run | Notes |
|---|---|---|
| 0.5B, R1/R1_accum, 5 gen, 2000 samples | ~1.0 hr | T4 fp16 |
| 0.5B, R2/R3, same | ~1.2 hr | extra real data |
| 0.5B, R4, same | ~1.5 hr | + teacher 1.5B 8-bit |
| 0.5B, Rn_*sweep* | ~1.3 hr avg | sweep cell |
| 1.5B, all regimes | ~3.0 hr avg | T4×2 (Kaggle) |

Total: 15 × 1.2 (0.5B main) + 15 × 1.3 (0.5B sweep) + 15 × 3.0 (1.5B main) ≈ **78 hrs**.

## Sequence (4 weekends)

### Weekend 1 — pre-reg + smoke
- [ ] Read `preregistration.md`, fill in any TBDs, commit, push, tag `sgt-prereg-v1`
- [ ] On Colab: run smoke cell (`R1 seed=11 generations=2 samples=200`) — confirms wiring in <15 min
- [ ] Check the smoke run JSON loads cleanly into `EarlyWarningAnalyzer.detect`
- [ ] Tag your `sgt_runner.py` commit hash; lock `requirements.lock.txt`

### Weekdays 1–2 — 0.5B main sweep (Colab, ~18 hrs)
- 5 regimes × 3 seeds = 15 runs
- One Colab session per regime (3 seeds back-to-back ≈ 3.5–4.5 hrs)
- Output: `drive/MyDrive/sgt_runs/0p5b/<regime>_<seed>.json`

### Weekend 2a — 0.5B correction sweep (Colab, ~7 hrs)
- 5 fractions × 3 seeds = 15 runs
- Single notebook session works; teacher = Qwen2.5-1.5B 8-bit
- Output: `…/0p5b/sweep/Rn_<pct>_<seed>.json`

### Weekend 2b — 1.5B main sweep (Kaggle, ~50 hrs across multiple sessions)
- 5 regimes × 3 seeds = 15 runs at ~3 hrs each
- Kaggle session cap is 12 hrs — batch ~4 runs/session
- Teacher for R4 = Qwen2.5-7B 4-bit (fits T4×2 with offload)
- Download the runs zip after each session, copy into the same Drive folder

### Weekend 3 — analysis (no GPU)
- [ ] `python sgt_analysis.py --runs runs/0p5b --runs runs/1p5b --out analysis/v1.json`
- [ ] Review `hypotheses` block; do not modify thresholds in response to results
- [ ] Build the v9.3 figure (3-seed median + 95% bootstrap band per regime)
- [ ] Write the v9.3 §4.5 "Pre-registration outcome" table directly from `analysis/v1.json`

### Weekend 4 — writeup + Zenodo
- [ ] Update v9.2 → v9.3 with replicated numbers
- [ ] Replace single-seed Table 1 with median + CI table; keep original as a footnote
- [ ] Push pre-reg + code to Zenodo as a separate dataset DOI
- [ ] File workshop submission (recommended: a NeurIPS 2026 collapse / reliable-ML workshop)

## Risks and mitigations (during execution)

| Risk | Mitigation |
|---|---|
| Colab kicks idle session | Keep a tiny `print` heartbeat every 5 min; resume from the last completed JSON, never overwrite |
| Kaggle internet off | Notebook settings → Internet ON before each session |
| Different transformers minor versions across sessions | `requirements.lock.txt` pinned; warn-and-fail in the runner if mismatch |
| 1.5B OOM on T4 | Drop batch_size to 4, gradient_accumulation 2; document in the analysis appendix |
| R4 teacher (Qwen2.5-7B) too slow | Fall back to teacher = Qwen2.5-1.5B (same as 0.5B run) and document the asymmetry |
| Discovering a bug after 1.5B is half done | Pre-reg §7 is explicit: re-run all affected configs from scratch with same seeds, no cherry-picking |

## Stop conditions (pre-committed)

1. If at end of Weekend 2b, ≤ 60% of runs complete: report partial v9.3 with a "compute-limited" caveat in §4.1; do not extend the design.
2. If a config crashes consistently across all 3 seeds: drop that regime from the table and explicitly call it "infeasible at this scale" in §6.
3. If two seeds agree and one disagrees by ≥ 2 generations on a key threshold: report all three, do not exclude.
