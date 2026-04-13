# PRISM

**Phase-based Research & Interpretability Spectral Microscope**

[![PyPI](https://img.shields.io/pypi/v/humanaiconvention-prism)](https://pypi.org/project/humanaiconvention-prism/)
[![Python](https://img.shields.io/pypi/pyversions/humanaiconvention-prism)](https://pypi.org/project/humanaiconvention-prism/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE)

PRISM is a model-agnostic mechanistic interpretability toolkit for transformer-family language models.  It measures **activation geometry** — the structural properties of hidden-state tensors that predict quantisation error, representation collapse, and alignment shifts — and exposes a 14-module suite of causal, spectral, and evaluation tools for deeper analysis.

---

## Why Activation Geometry?

When a language model is fine-tuned, its weights change.  What changes in the *activations* is harder to see — and more revealing.  PRISM's geometry metrics, inspired by [TurboQuant (Google, ICLR 2026)](https://arxiv.org/abs/2406.02544), measure four per-layer statistics that together capture how "hostile" a layer's activations are to low-bit quantisation:

| Metric | What it measures | Danger sign |
|---|---|---|
| `outlier_ratio` | max dim magnitude / mean dim magnitude | > 10 |
| `activation_kurtosis` | heavy-tail-ness of per-dim magnitudes | large positive |
| `cardinal_proximity` | how axis-aligned unit vectors are | near 1.0 |
| `quantization_hostility` | composite score in [0, 1] | > 0.70 |

A high `quantization_hostility` score means the layer will lose significant information when quantised to 4- or 8-bit precision.  Tracking this score **across training runs** reveals whether fine-tuning is moving a model toward or away from a quantisation-friendly (and typically more robust) geometry.

---

## Quickstart

```bash
pip install humanaiconvention-prism
```

**Measure the quantisation hostility of any model in 3 lines:**

```python
from prism.geometry import scan_model_geometry

results = scan_model_geometry("google/gemma-4-e2b-it")
print(results["mean_quantization_hostility"])   # e.g. 0.914
```

That is the complete API for the primary use case.  `scan_model_geometry` loads the model from the Hugging Face Hub, runs a single forward pass, and returns per-layer geometry metrics — no hook management, no manual tokenisation, no dtype juggling.

---

## Case Study: Gemma 4 E2B Fine-Tuning Geometry

The result that motivated PRISM's public release came from the **Gemma4Good Hackathon (April 2026)**, where the HumanAI Convention team fine-tuned Gemma 4 E2B using QLoRA on curated semantic-grounding interview data.

PRISM tracked `quantization_hostility` at three checkpoints:

| Checkpoint | Mean hostility | SGT score | Security failures |
|---|---|---|---|
| **Gemma 4 E2B baseline** | 0.9146 | — | — |
| **HAIC v1 adapter** (BEAST, untagged data) | 0.9144 | 6.20 / 10 | 0 |
| **HAIC v2 adapter** (A100, SGT-formatted data) | **0.7398** | **8.56 / 10** | **0** |

The v2 adapter achieved a **−0.175 reduction in mean quantisation hostility** — a 19% shift — while simultaneously producing the strongest SGT behavioural score in the HAIC model family.  This is not coincidence.

The geometry shift tracks the training-data quality change precisely:

- **v1** was trained on `grounding_mix_v3i.jsonl` — 975 examples with no `[PIVOT:]` format tags.  The LoRA adapted the model's style but never saw the target output format.  Hostility: unchanged (0.9144, Δ = −0.0002 — noise level).
- **v2** was trained on `grounding_gemma4_v2.jsonl` — 500 SGT-formatted examples with 100% `[PIVOT:]` coverage, three training windows per conversation (T2 + T4 + T6 loss), on an A100.  The geometry *shifted*.

**Interpretation under C(t):** LoRA adapters generally navigate the activation manifold rather than reshape it — which is why v1 showed geometry-stable Δ ≈ 0.  The v2 geometry shift from 0.9146 to 0.7398 indicates that the higher-quality, format-consistent training data moved the adapter outside the geometry-neutral zone.  The model is not just stylistically different; its residual-stream structure has measurably changed in a direction that reduces quantisation sensitivity and — empirically — correlates with improved task performance and maintained security.

**This is the kind of result that PRISM was built to surface.**  Without geometry tracking, the v1 and v2 adapters look similar on loss curves.  With it, the bifurcation point is obvious.

---

## Installation

```bash
# Standard (CPU / CUDA)
pip install humanaiconvention-prism

# With BitsAndBytes for 4-bit / 8-bit quantised models
pip install 'humanaiconvention-prism[quantized]'

# With notebook / visualisation tools
pip install 'humanaiconvention-prism[notebook]'

# Full install
pip install 'humanaiconvention-prism[all]'
```

Or from source:

```bash
git clone https://github.com/humanaiconvention/prism.git
cd prism && pip install -e .
```

---

## API Reference

### `scan_model_geometry` — model-level scan

```python
from prism.geometry import scan_model_geometry

# From a HF Hub model id (auto-loads model + tokenizer)
results = scan_model_geometry("google/gemma-4-e2b-it")

# With a pre-loaded model
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
results = scan_model_geometry(model, tokenizer=tokenizer)

# 4-bit quantised (requires bitsandbytes + accelerate)
results = scan_model_geometry("google/gemma-4-e2b-it", load_in_4bit=True)
```

**Return value** — a flat dict:

```python
{
    "model_name":                 str,    # identifier
    "prompt":                     str,    # probe text used
    "n_layers":                   int,    # transformer block count
    "layers":                     list,   # per-layer dicts (see below)
    "mean_quantization_hostility": float, # mean across all layers
    "worst_layer_idx":            int,
    "best_layer_idx":             int,
    "worst_layer_hostility":      float,
    "best_layer_hostility":       float,
    "n_hostile_layers":           int,    # layers with hostility > 0.70
}

# Each element of "layers":
{
    "layer_idx":             int,
    "outlier_ratio":         float,
    "activation_kurtosis":   float,
    "cardinal_proximity":    float,
    "quantization_hostility": float,
}
```

---

### `outlier_geometry` — single-layer, tensor input

```python
from prism.geometry import outlier_geometry
import torch

# hidden states from a single layer: (seq_len, hidden_dim)
H = torch.randn(64, 2048)
metrics = outlier_geometry(H)
# {'outlier_ratio': ..., 'activation_kurtosis': ...,
#  'cardinal_proximity': ..., 'quantization_hostility': ...}
```

Accepts both `torch.Tensor` and `numpy.ndarray`.

---

### `outlier_geometry_numpy` — pure-NumPy fallback

For environments where PyTorch is not available (lightweight CI, pre-computed
activation files, etc.):

```python
from prism.geometry import outlier_geometry_numpy
import numpy as np

H = np.load("layer_12_hidden_states.npy")   # (seq_len, hidden_dim)
metrics = outlier_geometry_numpy(H)
```

---

### Tracking geometry across training

```python
from prism.geometry import scan_model_geometry

checkpoints = [
    "path/to/checkpoint-500",
    "path/to/checkpoint-1000",
    "path/to/checkpoint-2000",
]

for ckpt in checkpoints:
    r = scan_model_geometry(ckpt)
    print(f"{ckpt}  hostility={r['mean_quantization_hostility']:.4f}  "
          f"hostile_layers={r['n_hostile_layers']}/{r['n_layers']}")
```

---

## Full Toolkit

PRISM is a 14-module library.  The geometry scanner is its primary entry point, but the full suite is available for deeper mechanistic analysis.

```python
from prism import SpectralMicroscope

microscope = SpectralMicroscope()
report = microscope.full_scan(model, tokenizer, prompt="The capital of France is")
# Returns: logit_lens, rank_profile, static_circuits,
#          attention_heatmap, positional_sensitivity, provenance
```

| Module | Import | Capability |
|---|---|---|
| Geometry scanner | `prism.geometry` | Quantisation hostility profiling |
| Causal patching | `prism.causal` | Activation swap & attribution patching |
| Logit / Tuned Lens | `prism.lens` | Vocabulary projection at every layer |
| Attention circuits | `prism.attention` | Induction head detection, OV/QK SVD |
| Linear probing | `prism.probing` | Concept directions, CKA drift |
| Sparse features | `prism.sae` | TopK SAE training & feature attribution |
| MLP decomposition | `prism.mlp` | Rank restoration, neuron mapping |
| Hybrid diagnostics | `prism.arch` | Recurrent / linear-attention architectures |
| Phase coherence | `prism.phase` | Hilbert phase, PLV, FFT spectral |
| Entropy dynamics | `prism.entropy` | Shannon / Rényi entropy profiles |
| Geometric viability | `prism.geometry` | Intrinsic dimensionality, Fisher curvature |
| Circuit discovery | `prism.discovery` | Automated circuit extraction |
| Evaluation | `prism.eval` | Calibration, drift, temporal collapse |
| Telemetry schemas | `prism.telemetry` | Verifiable snapshot & delta proofs |

---

## Architecture Compatibility

PRISM's architecture adapter resolves transformer components without hard-coding
any single layout.  Tested families include:

LLaMA · Gemma (2, 3, 4, E2B, E4B) · Qwen2 / Qwen3 · Mistral · Phi · GPT-2 / GPT-NeoX · Pythia · SmolLM2 · OLMo2 · Mamba / GLA / FoX hybrids · T5 · mT5

Any model that supports `output_hidden_states=True` works with `scan_model_geometry`.

---

## Runtime Notes

1. **Precision** — `float32` avoids spectral-decomposition instabilities on CUDA.  BitsAndBytes quantised activations are automatically cast to `float32` for metric computation.
2. **Memory** — a single forward pass stores all layer hidden states simultaneously.  For very large models, reduce sequence length via the `prompt` parameter.
3. **Attention hooks** — use `attn_implementation="eager"` if flash-attention drops hidden-state hooks on your architecture.

---

## Genesis-152M Research Suite

`experiments/genesis/` is a complete 12-phase mechanistic interpretability research run on [guiferrarib/genesis-152m-instruct](https://huggingface.co/guiferrarib/genesis-152m-instruct), a 152M-parameter hybrid GLA/FoX model.  It serves as a worked example of PRISM applied end-to-end.

```bash
python experiments/genesis/go.py --list   # list all phases
python experiments/genesis/go.py 10A      # run a phase
```

---

## Testing

```bash
pytest                        # full suite
pytest tests/test_geometry*   # geometry module only
pytest --cov=prism            # with coverage
```

---

## Contributing

Contributions expanding architecture coverage, adding new geometry metrics, or
improving the test suite are welcome.  All new modules must include tests under
`tests/`.  See `CONTRIBUTING.md` for guidelines.

---

## Citation

```bibtex
@software{prism_2026,
  title  = {PRISM: Phase-based Research \& Interpretability Spectral Microscope},
  author = {HumanAI Convention Contributors},
  year   = {2026},
  url    = {https://github.com/humanaiconvention/prism},
}
```

---

## License

[Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE)
