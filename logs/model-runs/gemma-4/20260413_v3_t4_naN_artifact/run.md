# PRISM Run — HAIC Gemma 4 E2B v3 (T4 NaN artifact — NOT CANONICAL)

Date: 2026-04-13 (Kaggle T4)
Script: prism.geometry.scan_model_geometry (merged bf16, device_map=auto)

## ⚠️ CAVEAT: This result is invalid for geometry comparison

T4 GPU experienced **34% step NaN collapse during adapter training**. When PRISM
computed activation geometry on the final merged checkpoint, those NaN-laden
activations were implicitly handled via `nan_to_num(0)`, causing widespread layer
dropout and geometry profile flattening.

**Result: `qh=0.0474` is a measurement artifact, not a true geometric property of
the v3 adapter.**

Canonical PRISM for v3 must be computed on Colab A100 once the adapter is
downloaded and re-run.

---

## Model

- Base: google/gemma-4-E2B-it
- Adapter: HAIC v3 QLoRA (Kaggle T4, v3 training variant)
- Merged: bf16, load_in_4bit=False
- **Status: NaN collapse occurred during training**

## Results (NOT CANONICAL)

| Metric | Value | Note |
|---|---|---|
| mean_quantization_hostility | 0.0474 | **INVALID — NaN artifact** |
| mean_outlier_ratio | 4.28 | Flattened by NaN zero-fill |
| mean_activation_kurtosis | ~0 | Collapsed |
| mean_cardinal_proximity | ~0 | Collapsed |
| worst_layer_idx | L1 | Unreliable |

## Diagnosis

T4 VRAM constraints during mixed-precision training caused repeated NaN
propagation in ~1 of 3 training steps. The final checkpoint contains layers with
dead activations (zeros from NaN conversion). When PRISM scanned the merged model,
these dead zones produced degenerate geometry metrics.

**The geometry profile is not comparable to v2 (A100, 0% NaN) or baseline (E2B,
0% NaN).**

## Next Step

**Canonical PRISM for v3 is pending.** The v3 adapter needs to be:
1. Downloaded from the Kaggle session's `gemma4-v3/adapter/best/` before session expires
2. Re-run on Colab A100 to capture clean geometry
3. Scanned with PRISM to produce the canonical result

Do not use the `qh=0.0474` result for any training-data quality or model-selection
decisions.

## Raw Files

See `prism_gemma4_v3.json` (Kaggle output, flagged as NaN artifact in ledger).
