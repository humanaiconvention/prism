# PRISM

**Phase-based Research & Interpretability Spectral Microscope**

Mechanistic interpretability and inline spectral telemetry for autoregressive language models. PRISM is a 14-module toolkit for dissecting what transformer-family models are doing, layer by layer, head by head, token by token — with a high-level scanning API and a full suite of causal, geometric, and evaluation tools underneath.

---

## Capabilities

### ⚡ One-Command Model Scan

```python
from prism import SpectralMicroscope

microscope = SpectralMicroscope()
report = microscope.full_scan(model, tokenizer, prompt="The capital of France is")
# Returns: logit lens trajectory, rank restoration profile,
# static circuit SVD, attention entropy heatmap, positional sensitivity.
```

---

### 1. Telemetry & Snapshot Schemas (`prism.telemetry`)

Structured capture and verification of internal model state.

- **PRISM Adapters** — native integration points for state snapshotting and analysis
- **Snapshot Logic** — extract and validate `EntropySnapshot` geometry configurations
- **Settlement Proofs** — compute `EntropyDeltaProof` and `GeometricHealthScore` for verifiable telemetry

### 2. Causal Intervention Suite (`prism.causal`)

Surgical tools for causal attribution and hypothesis testing.

- **Activation Patching** — swap activations at any residual position to isolate causal components; supports iterative test schedules
- **Attribution Patching (AtP)** — gradient-based approximation for fast causal screening without full forward-pass overhead
- **Knockout Circuits** — head-level ablation combined with spectral readouts

### 3. Automated Circuit Discovery (`prism.discovery`)

End-to-end extraction of functional subnetworks.

- **CircuitScout** — automated extraction of named circuits (e.g. IOI name-mover circuits)
- **Graph Export** — subnetwork connectivity exported to standard graph formats

### 4. Evaluation & Grounding Metrics (`prism.eval`)

Behavioral and geometric evaluation under distribution shift and across recursive training generations.

- **Semantic Grounding Metrics** — precision, recall, F1 for conceptual preservation (`GroundingMetrics`); both exact-match and dense-embedding modes
- **Calibration & Diversity** — Expected Calibration Error (ECE) and n-gram / self-BLEU diversity (`CalibrationMetrics`, `DiversityMetrics`)
- **Early Warning Detection** — detect silent semantic drift before perplexity rises; configurable threshold sensitivity (`EarlyWarningAnalyzer`, `EarlyWarningDetector`)
- **Geometric Drift Extension** — augment behavioral trajectories with PRISM spectral signals (`spectral_entropy`, `effective_dimension`) to catch mechanistic precursors early
- **Temporal Collapse Analysis** — compute T_OOD / T_PPL / Δt and classify failure regimes (accuracy-first, perplexity-first, synchronized, no-collapse) across recursive training generations (`TemporalAnalyzer`)

### 5. Hessian & Loss Landscape Diagnostics (`prism.geometry.hessian`)

Characterize the curvature of the loss landscape around a model checkpoint.

- **Failure Mode Classification** — detect distribution shift risk, hallucination susceptibility, and adversarial brittleness
- **Landscape Metrics** — hardware-aware extraction of spectral sharpness, condition numbers, and negative saddle points

### 6. Attention Circuit Analysis (`prism.attention`)

Decompose attention heads into functional roles via weight-space and dynamic analysis.

- **Induction Head Detection** — automated scoring of in-context learning circuits
- **Weight-Space SVD** — singular value analysis of $W_O W_V$ (copying) and $W_Q^T W_K$ (matching)
- **Attention Entropy Maps** — per-head aggregation heatmaps from live inference

### 7. Vocabulary Projection & Lens Suite (`prism.lens`)

Track semantic crystallization as information propagates through depth.

- **Logit Lens** — direct $W_U$ projection of hidden states at every layer
- **Tuned Lens** — learned affine translators for cleaner early-layer decoding
- **Prediction Entropy** — uncertainty trajectory across model depth

### 8. Linear Probing & Steering (`prism.probing`)

Identify and manipulate linear structure in the residual stream.

- **Concept Probing** — logistic regression probes to identify linear representation directions
- **Causal Steering** — extract Causal Inner Product (CIP) vectors for behavioral control
- **CKA Drift** — Centered Kernel Alignment tracking for geometric representation drift

### 9. Sparse Feature Decomposition (`prism.sae`)

Decompose polysemantic neurons into interpretable sparse features.

- **SAE Training** — TopK Sparse Autoencoders trained on arbitrary hidden states
- **Feature Attribution** — project learned sparse features back to vocabulary concepts

### 10. FFN / MLP Mechanistic Decomposition (`prism.mlp`)

Characterize how FFN layers process and transform representations.

- **Rank Restoration Profile** — quantify how MLP layers restore effective rank after attention compression
- **Key-Value Neuron Mapping** — map neurons to the tokens that most activate them

### 11. Hybrid & Cross-Architecture Diagnostics (`prism.arch`)

Tools built for non-standard architectures including linear attention and recurrent hybrids.

- **Recurrent Attractors** — track Principal Angles in GLA/Mamba-style recurrent state spaces
- **Positional Sensitivity** — measure drift and rank collapse under RoPE/ALiBi ablation
- **Linear vs. Softmax Fingerprinting** — compare spectral signatures across hybrid mixer types

### 12. Phase Synchronization & Spectral Coherence (`prism.phase`)

Treat hidden states as signals and measure their phase structure.

- **Hilbert Phase Extraction** — instantaneous phase angles along the token sequence
- **Cross-Layer PLV** — Phase Locking Values to identify coherent multi-layer processing
- **FFT Telemetry** — dominant spatial frequency identification in activation space
- **Phase Clustering** — circular variance to detect semantic grouping moments

### 13. Entropy Reduction Dynamics (`prism.entropy`)

Profile the model's decision process as it commits to a prediction.

- **Expansion vs. Pruning Profiles** — track $\Delta H$ to identify computational phases
- **Rényi Entropy Sweeps** — map probability mass consolidation with tunable sensitivity
- **Spectral–Semantic Coupling** — correlate geometric rank compression with prediction certainty
- **Entropy Phase Transitions** — detect commitment points via piecewise linear fitting

### 14. Viability Constraints & Geometric Validity (`prism.geometry`)

Assess the health and structure of the representation manifold.

- **Effective Dimension Viability Curve** — normalized geometric health scores across depth
- **ID Hunchback Profile** — characterize the rise and fall of intrinsic dimensionality
- **Representational Noise Sensitivity** — robustness margin via calibrated noise injection
- **Fisher Information Curvature** — representation manifold tightness and causal bottlenecks

---

## Installation

```bash
git clone https://github.com/humanaiconvention/prism.git
cd prism
pip install -e .
```

### Notebook environment (optional)

```bash
pip install -e ".[notebook]"
# or: pip install -r requirements.txt
```

### Dev / test environment

```bash
pip install -e ".[dev]"
```

---

## Quickstart

PRISM works with any Hugging Face `AutoModelForCausalLM` that supports `output_hidden_states=True`.

### Full mechanistic scan

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from prism import SpectralMicroscope

model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    attn_implementation="eager",   # required for hidden state hooks on some architectures
    torch_dtype=torch.float32,     # recommended for stable spectral decomposition
)

microscope = SpectralMicroscope()
report = microscope.full_scan(
    model, tokenizer,
    prompt="The capital of France is",
    target_layer=None,             # defaults to middle layer
)

# report keys: static_circuits, logit_lens, rank_profile,
#              attention_heatmap, positional_sensitivity, provenance
print(report["logit_lens"]["top_predictions"])
print(report["rank_profile"])
```

### Streaming telemetry

```python
microscope = SpectralMicroscope(max_tokens=64, window_size=32, streaming_cov_alpha=0.95)

result = microscope.generate_and_analyze(
    model=model,
    tokenizer=tokenizer,
    prompt="Explain what a black hole is in three simple sentences.",
    max_new_tokens=50,
)
```

### Individual modules

```python
from prism import (
    ActivationPatcher,        # causal patching
    AttentionAnalyzer,        # attention SVD + entropy
    LogitLens, TunedLens,     # vocabulary projection
    ConceptProber,            # linear probing
    SAETrainer,               # sparse autoencoders
    MLPAnalyzer,              # rank restoration
    PhaseAnalyzer,            # phase coherence
    EntropyDynamics,          # entropy profiles
    GeometricViability,       # manifold health
    HybridDiagnostics,        # recurrent / hybrid arch
)
```

---

## Runtime notes

1. **Attention implementation** — use `attn_implementation="eager"` if flash attention drops hidden state hooks.
2. **Numeric stability** — spectral decomposition is sensitive to precision; `float32` on CUDA avoids early-step NaNs.
3. **Memory** — telemetry stores layer-wise activations; reduce batch size relative to a standard generation workload.

---

## Genesis-152M Replication Suite

`experiments/genesis/` is a complete mechanistic interpretability research suite run on [guiferrarib/genesis-152m-instruct](https://huggingface.co/guiferrarib/genesis-152m-instruct), a 152M-parameter hybrid GLA/FoX model. It covers Phases 0A through 12F and serves as a worked example of PRISM applied to a real architecture.

**Selected findings:**

| Finding | Result |
|---|---|
| True effective rank at final layer | 185.5 / 576 (32.2%) — prior short-sequence measurements were artifacts |
| Compression bottleneck | Mixer (GLA/FoX), not residual add; ~4× ER oscillation per block |
| FFN role | Major rank restorer (+124 ER at L15) |
| GLA attractor convergence | Recurrent subspace locks at 46.2° relative to T=0 by step t=27 |
| Steering corridor | L7–L11 FoX band is the most responsive; OOD transfer is family-sensitive and fragile |
| L15 Lexical Crossover | Vocabulary ER collapses from >100 to ~28 words at the causal geometric bottleneck |
| Period-4 oscillation | Welford ER tracking and FFT both show a power spike at f=0.25, matching GLA/GLA/GLA/FoX layout |
| Circuit Closure (Phase 12) | L11 W_o is a high-gain output stage; the corridor is a non-specific answer-adjacent access interface |

```bash
# List all phases
python experiments/genesis/go.py --list

# Inspect a phase
python experiments/genesis/go.py --info 10A

# Run a phase
python experiments/genesis/go.py 10A
```

See `experiments/genesis/README.md` for the full phase-by-phase account.

---

## Testing

PRISM ships a comprehensive pytest suite covering all 14 modules:

| Module group | Test file(s) |
|---|---|
| Core telemetry schemas | `test_telemetry.py` |
| Adapter coverage | `test_adapter_coverage.py` |
| Attention circuits | `test_attention.py` |
| Causal patching | `test_causal.py` |
| Circuit discovery | `test_circuit_discovery.py` |
| Geometry / Hessian | `test_integration_advanced_geometry.py`, `test_hessian_diagnostics.py` |
| Logit / Tuned lens | `test_lens.py` |
| MLP decomposition | `test_mlp.py` |
| Probing & steering | `test_probing.py` |
| SAE features | `test_sae.py` |
| Evaluation metrics | `test_eval_metrics.py`, `test_eval_calibration.py` |
| Drift & early warning | `test_eval_early_warning.py`, `test_eval_drift_metrics.py` |
| Temporal collapse | `test_eval_temporal.py` |
| Hybrid / arch | `test_hybrid.py` |
| Hardware-aware analysis | `test_prism_analysis_hardware.py` |
| Integration / master scan | `test_integration_master_scan.py`, `test_integration_advanced_diagnostics.py` |
| Math proofs | `test_math_*.py` (10 files) |

```bash
pytest
```

---

## Model Run Logs

For bounded model analyses and comparison sweeps, use `scripts/run_model_analysis.py`. It writes a canonical PRISM log bundle under `logs/model-runs/` — one folder per run, containing the summary, findings, lessons, phase logs, and mirrored artifacts. Use `MODEL_RUN_TEMPLATE.md` as the standard format for recording findings.

---

## Contributing

Contributions to expand telemetry capabilities, add new architectures, or introduce new analytical metrics are welcome. All new modules must be covered by tests in `tests/`. See `CONTRIBUTING.md` for guidance.

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
