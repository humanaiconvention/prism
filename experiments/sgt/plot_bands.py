"""3-seed band figure for SGT v9.3 §4 Figure 2.

Reads per-(regime, seed) run JSONs from one or more --runs directories,
groups by regime, plots PPL and ACC trajectories with median line and
min/max envelope across seeds. Floor-censored runs are shown with hatching
on the ACC panel.

Usage:
    python plot_bands.py --runs runs/0p5b --out analysis/figure2_bands.png

The output is whatever matplotlib will write — a .png path produces a PNG,
a .pdf path produces a PDF. The script writes both side by side if you pass
either extension.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REGIME_ORDER = ["R1", "R1_accum", "R2", "R3", "R4"]
REGIME_COLOR = {
    "R1": "#d62728",        # red — closure / replace
    "R1_accum": "#ff7f0e",  # orange — closure / accumulate
    "R2": "#2ca02c",        # green — frozen real anchor
    "R3": "#1f77b4",        # blue — fresh real
    "R4": "#9467bd",        # purple — teacher correction
}
ACC_FLOOR = 0.20  # pre-reg §3 floor for ACC censoring


def load_runs(run_dirs: list[str]) -> list[dict]:
    runs = []
    for d in run_dirs:
        for jf in sorted(Path(d).glob("*.json")):
            try:
                runs.append(json.loads(jf.read_text()))
            except Exception as exc:
                print(f"!! could not parse {jf}: {exc}")
    return runs


def group_by_regime(runs: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        reg = r["cfg"]["regime"]
        if reg.startswith("Rn_"):
            continue  # correction sweep handled separately
        out[reg].append(r)
    return out


def history_to_arrays(history: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gens = np.array([h["generation"] for h in history], dtype=int)
    ppl = np.array([h["grounded_arc_perplexity"] for h in history], dtype=float)
    acc = np.array([h["grounded_arc_accuracy"] for h in history], dtype=float)
    return gens, ppl, acc


def plot(runs: list[dict], out_path: str) -> None:
    by_regime = group_by_regime(runs)
    fig, (ax_ppl, ax_acc) = plt.subplots(2, 1, figsize=(7.5, 7.0), sharex=True)

    plotted_regimes = []
    for regime in REGIME_ORDER:
        seed_runs = by_regime.get(regime, [])
        if not seed_runs:
            continue
        plotted_regimes.append(regime)

        # stack per-seed trajectories (assumes shared generation grid)
        gens = None
        ppls, accs, floor_flags = [], [], []
        for r in seed_runs:
            g, p, a = history_to_arrays(r["history"])
            if gens is None:
                gens = g
            ppls.append(p)
            accs.append(a)
            floor_flags.append(r["history"][0]["grounded_arc_accuracy"] < ACC_FLOOR)

        ppls = np.stack(ppls, axis=0)
        accs = np.stack(accs, axis=0)
        ppl_med = np.median(ppls, axis=0)
        ppl_lo, ppl_hi = ppls.min(axis=0), ppls.max(axis=0)
        acc_med = np.median(accs, axis=0)
        acc_lo, acc_hi = accs.min(axis=0), accs.max(axis=0)

        color = REGIME_COLOR[regime]
        n_seeds = len(seed_runs)
        n_floor = sum(floor_flags)

        ax_ppl.plot(gens, ppl_med, "-", color=color,
                    label=f"{regime} (n={n_seeds})", linewidth=2)
        ax_ppl.fill_between(gens, ppl_lo, ppl_hi, color=color, alpha=0.18)

        ax_acc.plot(gens, acc_med, "-", color=color, linewidth=2,
                    label=f"{regime} (n={n_seeds}, floor-censored {n_floor}/{n_seeds})")
        if n_floor < n_seeds:
            ax_acc.fill_between(gens, acc_lo, acc_hi, color=color, alpha=0.18)
        if n_floor > 0:
            ax_acc.fill_between(gens, acc_lo, acc_hi, facecolor="none",
                                edgecolor=color, hatch="//", alpha=0.35)

    ax_ppl.set_ylabel("Validation perplexity (WikiText-2)")
    ax_ppl.set_title("SGT v9.3 §4 Figure 2 — 0.5B 3-seed bands per regime")
    ax_ppl.grid(True, alpha=0.3)
    ax_ppl.legend(loc="best", fontsize=8)

    ax_acc.set_xlabel("Generation")
    ax_acc.set_ylabel("ARC-Easy accuracy (free-form, exact-match)")
    ax_acc.axhline(ACC_FLOOR, linestyle=":", color="grey", linewidth=1,
                   label=f"pre-reg §3 floor (ACC = {ACC_FLOOR})")
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(loc="best", fontsize=8)

    plt.tight_layout()

    out = Path(out_path)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    if out.suffix == ".png":
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    elif out.suffix == ".pdf":
        fig.savefig(out.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    print(f"regimes plotted: {plotted_regimes}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", action="append", required=True,
                    help="dir of per-run JSONs; pass once per scale or location")
    ap.add_argument("--out", default="analysis/figure2_bands.png")
    args = ap.parse_args()

    runs = load_runs(args.runs)
    if not runs:
        raise SystemExit("no runs found in any --runs dir")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plot(runs, args.out)


if __name__ == "__main__":
    main()
