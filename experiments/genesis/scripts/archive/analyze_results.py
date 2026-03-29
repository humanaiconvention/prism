"""Analyze 2-hour suite results."""
import csv
import numpy as np

def analyze_layer_geometry(path):
    rows = list(csv.DictReader(open(path, 'r', encoding='utf-8')))
    print(f"=== LAYER GEOMETRY ({len(rows)} prompts) ===\n")
    
    all_gla = []
    all_fox = []
    
    for row in rows:
        p = row['prompt_text_short'][:50]
        g = float(row['gla_mean_eff_rank'])
        f = float(row['fox_mean_eff_rank'])
        o = float(row['overall_eff_rank'])
        n = float(row['mean_nll'])
        gs = float(row['gla_std_eff_rank'])
        fs = float(row['fox_std_eff_rank'])
        gn = float(row['gla_mean_norm'])
        fn = float(row['fox_mean_norm'])
        all_gla.append(g)
        all_fox.append(f)
        print(f"  {p}")
        print(f"    GLA: {g:.1f}±{gs:.1f}  FoX: {f:.1f}±{fs:.1f}  Overall: {o:.1f}/576 ({o/576*100:.1f}%)  NLL: {n:.3f}")
        print(f"    Norms — GLA: {gn:.0f}  FoX: {fn:.0f}")
        print()
    
    print(f"  AGGREGATE:")
    print(f"    GLA mean EffRank: {np.mean(all_gla):.1f} ± {np.std(all_gla):.1f}")
    print(f"    FoX mean EffRank: {np.mean(all_fox):.1f} ± {np.std(all_fox):.1f}")
    print(f"    Δ(FoX - GLA): {np.mean(all_fox) - np.mean(all_gla):+.1f}")
    
    # Per-layer detail for first prompt
    print(f"\n  PER-LAYER DETAIL (Prompt 0):")
    row0 = rows[0]
    for i in range(30):
        ltype = "fox" if i in [3,7,11,15,19,23,27] else "gla"
        key = f"L{i}_{ltype}_eff_rank"
        er = float(row0[key])
        nkey = f"L{i}_{ltype}_norm"
        nm = float(row0[nkey])
        bar = "█" * int(er / 2)
        marker = " *** FoX" if ltype == "fox" else ""
        print(f"    L{i:2d} ({ltype}): ER={er:5.1f}  norm={nm:6.0f}  {bar}{marker}")

def analyze_spectral(path):
    rows = list(csv.DictReader(open(path, 'r', encoding='utf-8')))
    print(f"\n\n=== SPECTRAL PROFILE V2 ({len(rows)} rows) ===\n")
    
    # Group by prompt
    prompts = {}
    for row in rows:
        pid = int(row['prompt_idx'])
        if pid not in prompts:
            prompts[pid] = {'text': row['prompt_text'][:50], 'cat': row['category'], 'rows': []}
        prompts[pid]['rows'].append(row)
    
    for pid in sorted(prompts.keys()):
        p = prompts[pid]
        nlls = [float(r['nll']) for r in p['rows']]
        sers = [float(r['shannon_eff_rank']) for r in p['rows']]
        prs = [float(r['effective_dim']) for r in p['rows']]
        seds = [float(r['streaming_eff_dim']) for r in p['rows']]
        
        print(f"  Prompt {pid} ({p['cat']}): {p['text']}")
        print(f"    NLL: {np.mean(nlls):.3f}  Shannon ER: {np.mean(sers):.1f} (max {max(sers):.1f})  "
              f"PR: {np.mean(prs):.1f}  StreamDim: {np.mean(seds):.1f}")
        
        # Show last 5 tokens
        last5 = p['rows'][-5:]
        for r in last5:
            print(f"      step {r['step']:>3s}: SER={float(r['shannon_eff_rank']):6.1f}  "
                  f"PR={float(r['effective_dim']):5.1f}  stream={float(r['streaming_eff_dim']):5.1f}  "
                  f"NLL={float(r['nll']):5.3f}  token='{r['token']}'")
        print()


if __name__ == "__main__":
    analyze_layer_geometry("logs/layer_geometry_normal.csv")
    analyze_spectral("logs/spectral_profile_v2.csv")
