"""SGT analysis - pre-registered. Reads runs/<model>/*.json, scores H1-H4."""
from __future__ import annotations
import argparse, json, math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

def _load_early_warning():
    import importlib.util, os
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "..", "src", "prism", "eval", "early_warning.py"),
        os.path.join(here, "..", "..", "..", "src", "prism", "eval", "early_warning.py"),
    ]
    for path in candidates:
        path = os.path.normpath(path)
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location("early_warning", path)
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            return mod.EarlyWarningAnalyzer
    raise RuntimeError("could not locate prism/eval/early_warning.py")

EarlyWarningAnalyzer = _load_early_warning()

PPL_THRESH = 0.05
ACC_THRESH = 0.10
SEED_PASS = 2

def score_run(record):
    cfg = record["cfg"]; hist = record["history"]
    metrics = [{
        "generation": h["generation"],
        "grounded_arc_perplexity": h["grounded_arc_perplexity"],
        "grounded_arc_accuracy": h["grounded_arc_accuracy"],
    } for h in hist]
    res = EarlyWarningAnalyzer(acc_threshold=ACC_THRESH,
                               ppl_threshold=PPL_THRESH).detect(metrics)
    t_ppl = res["t_ppl"]; t_acc = res["t_acc"]
    if t_ppl == -1 and t_acc == -1: censor, dt = "both_censored", None
    elif t_ppl == -1:                censor, dt = "ppl_censored", None
    elif t_acc == -1:                censor, dt = "acc_censored", None
    else:                            censor, dt = "uncensored", t_ppl - t_acc
    floor_censored = (hist[0]["grounded_arc_accuracy"] < 0.20)
    return {
        "regime": cfg["regime"], "seed": cfg["seed"], "model": cfg["model"],
        "t_perplexity": t_ppl, "t_accuracy": t_acc, "delta_t": dt,
        "censoring": censor, "floor_censored": floor_censored,
        "signature_detected": res["signature_detected"],
        "baseline_ppl": hist[0]["grounded_arc_perplexity"],
        "baseline_acc": hist[0]["grounded_arc_accuracy"],
        "final_acc": hist[-1]["grounded_arc_accuracy"],
        "final_ppl": hist[-1]["grounded_arc_perplexity"],
    }

def bootstrap_ci(values, n_boot=1000, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng(0)
    arr = np.array([v for v in values if v is not None], dtype=float)
    if len(arr) == 0: return (math.nan, math.nan, math.nan)
    boots = np.array([rng.choice(arr, size=len(arr), replace=True).mean()
                      for _ in range(n_boot)])
    return (float(np.median(arr)), float(np.quantile(boots, alpha/2)),
            float(np.quantile(boots, 1 - alpha/2)))

def majority_pass(bools, k=SEED_PASS):
    return sum(1 for b in bools if b) >= k

def score_h1(scored, model):
    closed, corrected = [], []
    for r in scored:
        if r["model"] != model or r["floor_censored"]:
            continue
        # Both-censored corrected runs count as |Delta t|=0 (no decoupling).
        # Other censored runs are excluded.
        if r["regime"] in ("R1", "R1_accum"):
            if r["delta_t"] is not None:
                closed.append(abs(r["delta_t"]))
        elif r["regime"] in ("R2", "R4"):
            if r["delta_t"] is not None:
                corrected.append(abs(r["delta_t"]))
            elif r["censoring"] == "both_censored":
                corrected.append(0.0)
    closed_pass = majority_pass(d >= 1 for d in closed)
    corr_pass   = majority_pass(d <= 1 for d in corrected)
    return {"closed_decoupled_pass": closed_pass,
            "corrected_coupled_pass": corr_pass,
            "passes": closed_pass and corr_pass,
            "closed_n": len(closed), "corrected_n": len(corrected)}

def score_h2(scored, model):
    sweep = [r for r in scored if r["model"] == model and r["regime"].startswith("Rn_")]
    if not sweep: return {"passes": None, "note": "no sweep runs"}
    by_frac = defaultdict(list)
    for r in sweep:
        try:
            frac = int(r["regime"].split("_")[1])
        except Exception:
            continue
        by_frac[frac].append(r["final_acc"])
    if not by_frac: return {"passes": None}
    high = max(by_frac); low = min(by_frac)
    pass_count = sum(1 for h, l in zip(by_frac[high], by_frac[low]) if h > l)
    return {"high_frac": high, "low_frac": low,
            "high_acc_per_seed": by_frac[high],
            "low_acc_per_seed":  by_frac[low],
            "passes": pass_count >= SEED_PASS}

def score_h3(scored, model):
    r1 = [r["signature_detected"] for r in scored if r["model"] == model and r["regime"] == "R1"]
    r4 = [r["signature_detected"] for r in scored if r["model"] == model and r["regime"] == "R4"]
    fires_r1 = majority_pass(r1)
    clears_r4 = majority_pass(not s for s in r4)
    return {"r1_fires": fires_r1, "r4_clears": clears_r4,
            "passes": fires_r1 and clears_r4,
            "r1_n": len(r1), "r4_n": len(r4)}

def score_h4(scored, model):
    medians = {}
    for regime in ("R1", "R1_accum", "R2", "R3", "R4"):
        dts = [r["delta_t"] for r in scored
               if r["model"] == model and r["regime"] == regime
               and r["delta_t"] is not None and not r["floor_censored"]]
        if dts: medians[regime] = float(np.median(dts))
    has_pos = any(v > 0 for v in medians.values())
    has_neg_or_cens = (any(v < 0 for v in medians.values()) or
        any(r["censoring"] != "uncensored" for r in scored if r["model"] == model))
    return {"medians": medians, "passes": has_pos and has_neg_or_cens}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", action="append", required=True,
                    help="dir containing per-run JSONs; pass once per model scale")
    ap.add_argument("--out", default="analysis/v1.json")
    a = ap.parse_args()

    scored = []
    for d in a.runs:
        for jf in sorted(Path(d).glob("*.json")):
            rec = json.loads(jf.read_text())
            s = score_run(rec)
            s["_path"] = str(jf); scored.append(s)

    models = sorted({s["model"] for s in scored})
    rng = np.random.default_rng(0)

    delta_t_ci = {}
    for m in models:
        for regime in sorted({s["regime"] for s in scored if s["model"] == m}):
            dts = [s["delta_t"] for s in scored
                   if s["model"] == m and s["regime"] == regime
                   and s["delta_t"] is not None and not s["floor_censored"]]
            delta_t_ci[f"{m}|{regime}"] = bootstrap_ci(dts, rng=rng)

    out = {
        "thresholds": {"ppl": PPL_THRESH, "acc": ACC_THRESH, "seed_pass": SEED_PASS},
        "scored_runs": scored,
        "delta_t_ci_median_lo_hi": delta_t_ci,
        "hypotheses": {
            m: {"H1": score_h1(scored, m), "H2": score_h2(scored, m),
                "H3": score_h3(scored, m), "H4": score_h4(scored, m)}
            for m in models
        },
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {a.out}  (n={len(scored)} runs across {len(models)} model scales)")

if __name__ == "__main__":
    main()
