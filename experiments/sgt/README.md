# SGT Recursive Grounding Benchmark

Pre-registered evidence for the v9.3 revision of *Semantic Grounding and the Preservation of Information in Recursive Systems*. The runs here generate the per-generation logs that the existing PRISM `EarlyWarningAnalyzer` consumes.

## Layout

```
experiments/sgt/
├── preregistration.md        Locked design, thresholds, decision rules
├── run_plan.md                Compute envelope, sequence, risks
├── sgt_runner.py              One run = one (regime, seed, model). CLI script.
├── sgt_analysis.py            Reads runs/, scores H1–H4, bootstraps Δt CIs.
├── notebooks/
│   ├── colab_runner.ipynb     L4/A100 dispatcher for 0.5B
│   └── kaggle_runner.ipynb    T4×2 dispatcher for 1.5B
├── runs/                       Per-run JSONs (gitignored)
└── analysis/                   Output tables, figures (committed)
```

## One-line quick start (after the smoke test)

```bash
python sgt_runner.py --regime R1 --seed 11 --model Qwen/Qwen2.5-0.5B --out runs/0p5b
```

Each run writes `runs/<out>/<regime>_<seed>.json` with the per-generation history dict. Format:

```json
{
  "cfg": {...},
  "history": [
    {"generation": 0, "grounded_arc_perplexity": 8.16, "grounded_arc_accuracy": 0.21},
    {"generation": 1, "grounded_arc_perplexity": 8.91, "grounded_arc_accuracy": 0.20},
    ...
  ],
  "baseline": {"ppl": 8.16, "acc": 0.21},
  "wallclock_s": 4180.7
}
```

This is the exact shape `prism.eval.early_warning.EarlyWarningAnalyzer.detect` expects.

## Reproducibility

- Pre-registration is committed at `preregistration.md` and tagged `sgt-prereg-v1` in git.
- Seeds are locked: 11, 23, 42.
- Thresholds are locked: PPL +5%, ACC −10%, majority pass = ≥2 of 3 seeds.
- Models, eval anchors, generation length, and decoding params all in the runner CLI defaults.
- After the sweep completes, `sgt_analysis.py` produces `analysis/v1.json` deterministically (seeded bootstrap).

## Reporting in v9.3

Final §4 numbers come from `analysis/v1.json`, not from any spreadsheet or memory. The §4.5 "Pre-registration outcome" table is generated directly from `analysis/v1.json["hypotheses"]`. Any number that doesn't appear in that file should not appear in the paper.
