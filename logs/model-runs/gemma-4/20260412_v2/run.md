# PRISM Run — HAIC Gemma 4 E2B v2 (canonical fine-tune)

Date: 2026-04-12 (Colab A100, kaggle-gemma4-v2)
Script: prism.geometry.scan_model_geometry (merged bf16 checkpoint, device_map=auto)

## Model

- Base: google/gemma-4-E2B-it
- Adapter: HAIC v2 QLoRA (grounding_gemma4_v2.jsonl, 500 examples, 100% PIVOT-tagged)
- Training hardware: Colab A100
- Merged: bf16, load_in_4bit=False (methodology-consistent with v1 and baseline)

## Results

| Metric | Value | Baseline (E2B) | Delta |
|---|---|---|---|
| mean_quantization_hostility | **0.7398** | 0.9146 | **−0.1748** |
| mean_outlier_ratio | — | 83.03 | — |
| mean_activation_kurtosis | — | 1009.34 | — |
| mean_cardinal_proximity | — | 0.7655 | — |
| SGT score | 8.56 / 10 | — | — |
| Security failures | 0 | — | — |

## Interpretation

First HAIC adapter to show a meaningful geometry shift. The −0.1748 delta on
quantization_hostility (19% reduction from baseline) correlates directly with the
training data quality change: v2 used 500 SGT-formatted examples with 100%
[PIVOT:] tag coverage and a multi-turn loss window (T2+T4+T6), vs v1's 975
untagged examples that produced Δ ≈ −0.0002 (noise level).

Under the C(t) framework: LoRA typically navigates the activation manifold without
reshaping it. The v2 geometry shift indicates the higher-quality, format-consistent
data moved the adapter outside the geometry-neutral zone — the residual stream
structure has measurably changed in a direction that reduces quantisation sensitivity
and correlates with improved task performance (SGT 8.56 vs v1's 6.20).

This result validates PRISM's use as a training-data quality signal, not just a
post-hoc checkpoint diagnostic.

## GGUF Notes

Local Q5_K_M (2026-04-12): Built on BEAST from Q8_0 via llama-quantize
--allow-requantize --tensor-type per_layer_token_embd.weight=q8_0.
Size: 4.173 GB (7.18 BPW). PLE tensor held at q8_0 (9.4 GB float32
allocation not feasible on Windows heap). GGUF geometry = Colab A100
canonical: qh=0.7398. Tail SHA256: 63e880619c38045f2c58b0a693d097f2cf1dea140b56a2bcdfb89c6a9a4402e6

## Full JSON

See result.json in this directory (Colab A100 run output).
