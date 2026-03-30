import csv
import sys

csv_path = sys.argv[1] if len(sys.argv) > 1 else "smoke_spectral.csv"
rows = list(csv.DictReader(open(csv_path, "r", encoding="utf-8")))
print(f"Total rows: {len(rows)}")
print(f"Columns: {list(rows[0].keys())}")
print()

prompts = sorted(set(r["prompt_idx"] for r in rows))
for p in prompts:
    prows = [r for r in rows if r["prompt_idx"] == p]
    print(f"Prompt {p} ({prows[0]['prompt_text'][:60]}...): {len(prows)} steps")
print()

print("=== First prompt, first 10 steps ===")
p0 = [r for r in rows if r["prompt_idx"] == "0"]
for r in p0[:10]:
    print(f"  step={r['step']:>3s}  NLL={float(r['nll']):6.3f}  "
          f"entropy={float(r['spectral_entropy']):6.3f}  "
          f"eff_dim={float(r['effective_dim']):7.2f}  "
          f"stream_dim={float(r['streaming_eff_dim']):7.2f}  "
          f"angle={float(r['projection_angle']):6.3f}  "
          f"norm={float(r['hidden_norm']):6.1f}")

print()
print("=== Per-prompt summary ===")
for p in prompts:
    prows = [r for r in rows if r["prompt_idx"] == p]
    nlls = [float(r["nll"]) for r in prows]
    ents = [float(r["spectral_entropy"]) for r in prows]
    edims = [float(r["effective_dim"]) for r in prows]
    sdims = [float(r["streaming_eff_dim"]) for r in prows]
    angs = [float(r["projection_angle"]) for r in prows]
    norms = [float(r["hidden_norm"]) for r in prows]
    print(f"Prompt {p}:")
    print(f"  NLL:          mean={sum(nlls)/len(nlls):.3f}  min={min(nlls):.3f}  max={max(nlls):.3f}")
    print(f"  Entropy:      mean={sum(ents)/len(ents):.3f}  min={min(ents):.3f}  max={max(ents):.3f}")
    print(f"  Eff dim:      mean={sum(edims)/len(edims):.1f}  min={min(edims):.1f}  max={max(edims):.1f}")
    print(f"  Stream dim:   mean={sum(sdims)/len(sdims):.1f}  min={min(sdims):.1f}  max={max(sdims):.1f}")
    print(f"  Proj angle:   mean={sum(angs)/len(angs):.3f}  min={min(angs):.3f}  max={max(angs):.3f}")
    print(f"  Hidden norm:  mean={sum(norms)/len(norms):.1f}  min={min(norms):.1f}  max={max(norms):.1f}")
