# Gemma 4 E4B vs E2B — PRISM Outlier-Geometry Analysis

**Date:** 2026-04-08  
**Method:** PRISM 20-prompt suite, bf16 (no quantization), `Gemma4ForConditionalGeneration`  
**E2B run:** `gemma4_e2b_baseline_verify.json` (reproducible baseline, confirmed 2026-04-08)  
**E4B run:** `gemma4_e4b_prism.json` (new, 2026-04-08)

---

## Aggregate Comparison

| Metric | E2B (2.0B) | E4B (4.0B) | Delta | Direction |
|---|---|---|---|---|
| quantization_hostility | 0.9146 | 0.9211 | +0.0065 | worse |
| outlier_ratio | 83.19 | 137.22 | +54.03 | worse |
| activation_kurtosis | 1009.57 | 1651.75 | +642.18 | worse |
| cardinal_proximity | 0.7660 | 0.7762 | +0.0102 | worse |
| worst_layer_idx | L29 | L2 | — | changed |
| best_layer_idx | L14 | L42 | — | changed |

**Summary:** Scaling from E2B to E4B worsens all four metrics. The outlier_ratio increase (+65%) is the most dramatic — E4B has significantly more extreme activation dimensions. This is consistent with the general finding that deeper/wider transformers develop stronger outlier channels.

---

## Architecture Reference

Gemma 4 uses a hybrid attention pattern (same across E2B and E4B):
- **Full-attention (FA) layers:** every 5th layer (0-indexed: 4, 9, 14, 19, 24, 29, 34, 39 in E4B)
- **Sliding-window layers:** all others (window=512 tokens)
- **E2B:** 35 layers (FA at L4, L9, L14, L19, L24, L29, L34)
- **E4B:** 43 layers (FA at L4, L9, L14, L19, L24, L29, L34, L39)
- Per-layer embeddings (PLE, 256-dim) at every layer — novel to Gemma 4
- Dual RoPE: sliding layers θ=10k, full-attention layers θ=1M

---

## Per-Layer Analysis

### Key findings from layer profile:

**L0 (embedding output):** hostility=0.167 — unusually clean. PLE normalization at input.

**L2 — worst layer in E4B (hostility=0.9850, outlier=353.5)**  
This is a sliding-window layer (not FA). The extreme outlier ratio (4.2× the E4B mean) appears at the third decoder layer, coinciding with the first major cross-attention accumulation. E2B shows the same region is elevated but not the worst (L2 E2B delta: +0.0955). The L2 spike may reflect where PLE contributions first destabilize the residual stream at the 4B scale.

**L3 — second worst in early zone (hostility=0.9745, outlier=219.5, delta=+0.217 vs E2B)**  
Immediately after L2, still pre-FA. The +0.217 delta is the third-largest positive delta in the profile. The L2-L3 cluster suggests a model-size-dependent instability in the early decoder that doesn't resolve before the first full-attention gate.

**L4 — first full-attention boundary (hostility=0.9626, delta=+0.0546)**  
FA layers were hypothesized to be worst. They are not — FA layers consistently score *better* than their adjacent sliding-window layers in both models. The global-context aggregation at FA layers appears to partially regularize outlier dimensions rather than amplify them.

**L9 (FA) and L19 (FA) — notable drops (delta: -0.102, -0.101)**  
Both show the largest negative deltas (E4B better than E2B at these positions). These are the middle full-attention layers where the information bottleneck is strongest. This pattern was also seen in E2B's layer analysis: FA layers are relatively low-hostility compared to surrounding SW layers.

**L14 (FA) — exception: large positive delta (+0.299)**  
L14 is the one full-attention layer where E4B is significantly *worse* than E2B. L14 is the midpoint of both models' depth (7/14 in E2B, 7/21 in E4B first half). This may be where the E4B's additional capacity introduces a new dominant outlier subspace that E2B's smaller weight matrices don't generate.

**L18-L19 — local minimum cluster (hostility ~0.800)**  
Both E2B and E4B show this "valley" around L18-L19. In E2B this is near the worst layer (L29); in E4B these layers are well below the mean. The L19 FA layer anchors the valley. This may be the functional sweet spot of the hybrid attention stack.

**L35-L41 — terminal ramp (E4B only)**  
The eight additional layers in E4B (L36-L43 relative to E2B's depth) show a sustained hostility plateau around 0.980-0.984 with outlier_ratio 167-212. This is the region unique to E4B with no E2B comparison. The plateau suggests the additional depth does not "resolve" the outlier geometry — it maintains it at a high steady state.

**L42 — best layer in E4B (hostility=0.569, outlier=10.9)**  
The terminal layer shows a dramatic collapse in hostility and outlier ratio — same pattern seen in E2B's L35 (terminal FA, the "collapse" we documented previously). This is the final full-attention layer (L39 is FA; L42 is the penultimate regular layer before output). The sharp drop is likely a normalization artifact at the model head boundary.

---

## Hypothesis Assessment

**Original hypothesis:** Full-attention boundary layers are the worst.  
**Result:** FALSIFIED (same conclusion as E2B analysis). FA layers are not the worst — they tend to be slightly *better* than adjacent sliding-window layers. The worst layers in E4B (L2, L3) are sliding-window layers in the early decoder.

**New finding — scaling hypothesis:** Outlier ratio scales superlinearly with model depth. E4B (+65% outlier ratio vs E2B) despite only a ~2× parameter increase suggests the outlier channel problem is amplified by depth more than by width alone. The L2-L3 spike pattern may be a scaling artifact specific to Gemma 4's PLE mechanism.

**Quantization implication:** E4B's outlier_ratio=137 (vs E2B=83) means Q4 quantization would degrade E4B more severely than E2B. For deployment, E4B should be quantized at Q5_K_M minimum; Q4_K_M may produce quality loss at L2-L3 due to the extreme outlier cluster.

---

## Model Arena Update

E4B data ready to add to `SpectralAnalysis.tsx` HAIC arena:

```typescript
{ id: 'gemma4-e4b', label: 'Gemma 4 E4B-it', family: 'Gemma', params: '4.0B',
  license: 'Gemma', hf_id: 'google/gemma-4-E4B-it',
  hostility: 0.9211, outlier_ratio: 137.2, activation_kurtosis: 1651.8,
  cardinal_proximity: 0.7762, worst_layer_zone: 'early',
  data_status: 'verified', run_date: '2026-04-08' },
```

---

## Files

- `data/model-runs/gemma4_e4b_prism.json` — raw E4B PRISM output (43 layers, 20 prompts)
- `data/model-runs/gemma4_e2b_baseline_verify.json` — E2B baseline (confirmed reproducible)
- `data/model-runs/gemma4_e2b_layer_analysis.json` — prior detailed E2B layer work
