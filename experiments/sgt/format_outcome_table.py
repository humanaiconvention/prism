"""Render the §4.5 Pre-registration outcome table from analysis/v1.json.

Usage:
    python format_outcome_table.py --in analysis/v1.json --out analysis/v9_3_section_4_5.md

Emits a Markdown table with H1-H4 results per model scale and an overall
"supported / partial / unsupported" call. Also prints a per-regime PPL/Δt
summary suitable for the §4.3 narrative paragraph and a one-line headline
for the abstract.

This is meant to be machine-mechanical: every cell traces to a key in the
analysis JSON. Read the script if you doubt a cell.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def fmt_ci(triple: list | tuple, places: int = 2) -> str:
    if not triple:
        return "n/a"
    med, lo, hi = triple
    if any(isinstance(x, float) and math.isnan(x) for x in (med, lo, hi)):
        return "n/a"
    return f"{med:.{places}f} [{lo:.{places}f}, {hi:.{places}f}]"


def call(passes: bool | None) -> str:
    if passes is True:
        return "PASS"
    if passes is False:
        return "FAIL"
    return "n/a"


def overall_call(per_scale: dict[str, bool | None]) -> str:
    """supported = both scales pass; partial = one scale; unsupported = neither."""
    vals = [v for v in per_scale.values() if v is not None]
    if not vals:
        return "n/a (no data)"
    if all(v is True for v in vals) and len(vals) >= 2:
        return "**supported**"
    if any(v is True for v in vals):
        return "**partial**"
    return "**unsupported**"


def render(analysis: dict) -> str:
    hyp = analysis.get("hypotheses", {})
    models = list(hyp.keys())
    out_lines: list[str] = []

    out_lines.append("## §4.5 Pre-registration outcome\n")
    out_lines.append("Decision rules verbatim from `experiments/sgt/preregistration.md` §4 at "
                     "tag `sgt-prereg-v1` (commit `6092c525`).\n")

    # Header
    cols = ["Hypothesis", "Decision rule (abbrev)"]
    for m in models:
        cols.append(f"At {m.split('/')[-1]}")
    cols.append("Combined call")
    out_lines.append("| " + " | ".join(cols) + " |")
    out_lines.append("|" + "|".join(["---"] * len(cols)) + "|")

    decision_rules = {
        "H1": "(R1, R1_accum) median \\|Δt\\| ≥ 1 in ≥2/3 seeds AND (R2, R4) median \\|Δt\\| ≤ 1 in ≥2/3 seeds. Both-censored R4 = \\|Δt\\|=0.",
        "H2": "Correction-sweep ARC at 75–100% > ARC at 0–25% in ≥2/3 seeds.",
        "H3": "EarlyWarningAnalyzer fires on R1 in ≥2/3 seeds AND clears on R4 in ≥2/3 seeds.",
        "H4": "At least one regime median Δt > 0 AND at least one < 0 (or censored) across the 5 regimes.",
    }
    for h in ("H1", "H2", "H3", "H4"):
        per_scale = {}
        cells = [h, decision_rules[h]]
        for m in models:
            entry = hyp[m].get(h, {})
            passes = entry.get("passes")
            per_scale[m] = passes
            detail = ""
            if h == "H1" and "closed_n" in entry:
                detail = f" (closed_n={entry['closed_n']}, corrected_n={entry['corrected_n']})"
            elif h == "H3" and "r1_n" in entry:
                detail = f" (R1 fires={entry['r1_fires']}, R4 clears={entry['r4_clears']})"
            elif h == "H4" and entry.get("medians"):
                detail = f" (medians={entry['medians']})"
            cells.append(call(passes) + detail)
        cells.append(overall_call(per_scale))
        out_lines.append("| " + " | ".join(str(c) for c in cells) + " |")

    out_lines.append("\n---\n")
    out_lines.append("### Per-regime Δt distribution (for §4.3 Table 1)\n")
    out_lines.append("Median Δt with bootstrap 95% CI, n=3 seeds, RNG seed=0, n_boot=1000.\n")

    cis = analysis.get("delta_t_ci_median_lo_hi", {})
    by_model: dict[str, dict[str, str]] = defaultdict(dict)
    for key, triple in cis.items():
        if "|" not in key:
            continue
        m, regime = key.split("|", 1)
        by_model[m][regime] = fmt_ci(triple)

    for m in models:
        out_lines.append(f"\n**{m.split('/')[-1]}**\n")
        out_lines.append("| Regime | Median Δt [95% CI] |")
        out_lines.append("|---|---|")
        for regime in ("R1", "R1_accum", "R2", "R3", "R4"):
            ci = by_model.get(m, {}).get(regime, "n/a")
            out_lines.append(f"| {regime} | {ci} |")

    out_lines.append("\n---\n")
    out_lines.append("### Per-regime narrative seeds (for §4.3 prose)\n")

    scored = analysis.get("scored_runs", [])
    by_regime_model: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for s in scored:
        by_regime_model[(s["model"], s["regime"])].append(s)

    for m in models:
        out_lines.append(f"\n**{m.split('/')[-1]}**\n")
        for regime in ("R1", "R1_accum", "R2", "R3", "R4"):
            rs = by_regime_model.get((m, regime), [])
            if not rs:
                continue
            n = len(rs)
            n_floor = sum(1 for r in rs if r.get("floor_censored"))
            n_sig = sum(1 for r in rs if r.get("signature_detected"))
            n_both_cens = sum(1 for r in rs if r.get("censoring") == "both_censored")
            n_acc_cens = sum(1 for r in rs if r.get("censoring") == "acc_censored")
            n_ppl_cens = sum(1 for r in rs if r.get("censoring") == "ppl_censored")
            dts = [r.get("delta_t") for r in rs if r.get("delta_t") is not None]
            dt_str = (f"median Δt = {sorted(dts)[len(dts)//2]:+d}" if dts else "all Δt censored")
            out_lines.append(
                f"- **{regime}**: n={n} seeds; "
                f"signature_detected = {n_sig}/{n}; "
                f"floor_censored = {n_floor}/{n}; "
                f"censoring = both:{n_both_cens} acc:{n_acc_cens} ppl:{n_ppl_cens}; "
                f"{dt_str}.")

    out_lines.append("\n---\n")
    out_lines.append("### One-line headline (for abstract)\n")

    # Best-effort headline
    headline_parts = []
    if models:
        m0 = models[0]
        h = hyp.get(m0, {})
        h1 = h.get("H1", {})
        h3 = h.get("H3", {})
        if h1.get("passes") is True:
            headline_parts.append(
                "perplexity-first decoupling is observed in closure regimes (R1, R1_accum) "
                "but not in adequately corrected regimes (R2, R4)")
        elif h1.get("passes") is False:
            headline_parts.append(
                "the decoupling-by-regime contrast (H1) does not pass at 0.5B; "
                f"closed_n={h1.get('closed_n')}, corrected_n={h1.get('corrected_n')}")
        if h3.get("passes") is True:
            headline_parts.append(
                "PRISM EarlyWarningAnalyzer separates R1 from R4 as predicted (H3 supported)")
        elif h3.get("passes") is False:
            headline_parts.append(
                "PRISM EarlyWarningAnalyzer does not separate R1 from R4 (H3 unsupported)")
    if not headline_parts:
        headline_parts.append("see §4.5 for outcomes (analysis returned no hypotheses block)")
    out_lines.append("\n> " + "; ".join(headline_parts) + ".")

    out_lines.append("")
    return "\n".join(out_lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="analysis/v1.json")
    ap.add_argument("--out", default="analysis/v9_3_section_4_5.md")
    args = ap.parse_args()

    analysis = json.loads(Path(args.inp).read_text())
    md = render(analysis)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md)
    print(md)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
