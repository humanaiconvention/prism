# Spectral Microscope

Inline spectral telemetry for autoregressive language-model generation.

## Features & Capabilities

PRISM provides a comprehensive mechanistic interpretability and telemetry framework organized into functional groups. All modules are fully implemented and optimized for local Windows hardware.

### ⚡ Automated Model Scanning
Run a multi-dimensional mechanistic report with a single command:
```python
microscope = SpectralMicroscope()
report = microscope.full_scan(model, tokenizer, prompt="The capital of France is")
# Returns: Logit Lens trajectory, Rank Restoration profile,
# Static Circuit SVD, and Positional Sensitivity metrics.
```

### 1. Telemetry & Snapshot Schemas (`prism.telemetry`)
*   **PRISM Adapters**: Native integration points for state snapshotting and analysis.
*   **Snapshot Logic**: Extract and validate `EntropySnapshot` geometry configurations.
*   **Settlement Proofs**: Calculate `EntropyDeltaProof` and `GeometricHealthScore` for verifiable telemetry.

### 2. Causal Intervention Suite (`prism.causal`)
*   **Activation Patching**: Surgical activation swaps to localize causal components (extended with iterative test support).
*   **Attribution Patching (AtP)**: Gradient-based approximation for fast causal screening.
*   **Knockout Circuits**: Head-level ablation combined with spectral readouts.

### 3. Automated Circuit Discovery (`prism.discovery`)
*   **CircuitScout**: Automated, end-to-end extraction of functional subnetworks (e.g. IOI name-mover circuits).
*   **Graph Formatting**: Direct export of subnetwork connectivity to standard graph reporting formats.

### 4. Evaluation & Grounding Metrics (`prism.eval`)
*   **Semantic Grounding Metrics**: Precision, recall, and F1 for conceptual preservation (`GroundingMetrics`). Dual exact-match and dense-embedding semantic similarity modes.
*   **Calibration & Diversity**: Expected Calibration Error (ECE) and n-gram / self-BLEU diversity scores (`CalibrationMetrics`, `DiversityMetrics`).
*   **Early Warning Detection**: Detect silent semantic drift — whether grounding failures precede perplexity rises — with configurable threshold sensitivity (`EarlyWarningAnalyzer`, `EarlyWarningDetector`).
*   **Geometric Drift Extension**: Augments behavioral trajectories with PRISM spectral signals (`spectral_entropy`, `effective_dimension`) to catch mechanistic precursors before behavioral metrics degrade.
*   **Temporal Collapse Analysis**: Compute T_OOD / T_PPL / Δt and classify failure regimes (accuracy-first, perplexity-first, synchronized, no-collapse) across recursive training generations (`TemporalAnalyzer`).

### 5. Hessian & Loss Landscape Diagnostics (`prism.geometry.hessian`)
*   **Failure Mode Classification**: Detect distribution shift risks, hallucination susceptibility, and adversarial brittleness.
*   **Landscape Metrics**: Hardware-aware extraction of spectral sharpness, condition numbers, and negative saddle points.

### 6. Attention Circuit Analysis (`prism.attention`)
*   **Induction Head Detection**: Automated scoring of in-context learning circuits.
*   **Weight-Space SVD**: Singular value analysis of $W_O W_V$ (Copying) and $W_Q^T W_K$ (Matching).
*   **Attention Entropy Maps**: Dynamic heatmap of head-level aggregation.

### 7. Iterative Inference & Vocabulary Projection (`prism.lens`)
*   **Logit Lens**: Direct $W_U$ projection to observe semantic crystallization.
*   **Tuned Lens**: Learned affine translators for 10x cleaner early-layer decoding.
*   **Prediction Entropy**: Tracking prediction uncertainty across model depth.

### 8. Linear Probing & Steering (`prism.probing`)
*   **Concept Probing**: Train logistic regression probes to identify linear directions.
*   **Causal Steering**: Extract Causal Inner Product (CIP) vectors for behavioral control.
*   **CKA Drift**: Measure Centered Kernel Alignment to track geometric representation drift.

### 9. Sparse Feature Decomposition (`prism.sae`)
*   **SAE Training**: TopK Sparse Autoencoders to decompose polysemantic neurons.
*   **Feature Attribution**: Projecting learned sparse features back to vocabulary concepts.

### 10. FFN / MLP Mechanistic Decomposition (`prism.mlp`)
*   **Rank Restoration Profile**: Quantify how MLP layers restore rank after attention compression.
*   **Key-Value Mapping**: Map specific neurons to the tokens that activate them.

### 11. Hybrid & Cross-Architecture Diagnostics (`prism.arch`)
*   **Recurrent Attractors**: Track Principal Angles in linear recurrent states (GLA/Mamba).
*   **Positional Sensitivity**: Measure drift and rank collapse under RoPE/ALiBi ablation.
*   **Linear vs. Softmax Fingerprinting**: Compare spectral signatures across hybrid mixers.

### 12. Phase Synchronization & Spectral Coherence (`prism.phase`)
*   **Hilbert Phase Extraction**: Extract instantaneous phase angles along the token sequence.
*   **Cross-Layer Phase Coherence (PLV)**: Compute Phase Locking Values to identify coherent multi-layer processing.
*   **FFT Telemetry**: Dominant spatial frequency identification in representations.
*   **Phase Clustering**: Measure circular variance to detect semantic grouping moments.

### 13. Entropy Reduction Dynamics (`prism.entropy`)
*   **Expansion vs. Pruning Profiles**: Track $\Delta H$ to identify model decision phases.
*   **Rényi Entropy Sweeps**: Map probability mass consolidation with tunable sensitivity.
*   **Spectral–Semantic Coupling**: Correlate geometric rank compression with prediction certainty.
*   **Entropy Phase Transitions**: Detect computational commitment points via piecewise linear fitting.

### 14. Viability Constraints & Geometric Validity (`prism.geometry`)
*   **Effective Dimension Viability Curve**: Normalized geometric health scores for model assessment.
*   **ID Hunchback Profile**: Characterize the rise and fall of intrinsic dimensionality across depth.
*   **Representational Noise Sensitivity**: Quantify robustness margin via calibrated noise injection.
*   **Fisher Information Curvature**: Measure representation manifold tightness and causal bottlenecks.

---

## Installation

```bash
git clone https://github.com/humanaiconvention/prism.git
cd prism
pip install -e .
```

### Notebook Environment (Optional)

```bash
pip install -e ".[notebook]"
# or: pip install -r requirements.txt
```

## Quickstart Usage

PRISM works with standard Hugging Face models by leveraging `output_hidden_states=True`. 

### Python API

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from prism import SpectralMicroscope, __version__

print(f"PRISM version: {__version__}")

# 1. Load any HF Model
model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    attn_implementation="eager", # Required for hidden states on some architectures
    torch_dtype=torch.float32    # Recommended for stable spectral decomposition
)

# 2. Attach Microscope
microscope = SpectralMicroscope(max_tokens=64, window_size=32, streaming_cov_alpha=0.95)

# 3. Generate and Analyze
result = microscope.generate_and_analyze(
    model=model,
    tokenizer=tokenizer,
    prompt="Explain what a black hole is in three simple sentences.",
    max_new_tokens=50
)

# 4. Inspect Telemetry
import pandas as pd
df = pd.DataFrame(result["telemetry"])
print(df.head())
```

See `quickstart.ipynb` for a fully interactive version with visualizations.

### Example Telemetry Schema

Per generated token, default telemetry includes:

| step | token | spectral_entropy | effective_dim | streaming_eff_dim | projection_angle |
|---:|---|---:|---:|---:|---:|
| 1 | `The` | 2.08 | 7.42 | 1.00 | 0.74 |
| 2 | ` event` | 2.21 | 8.03 | 1.88 | 0.71 |
| 3 | ` horizon` | 2.33 | 8.67 | 2.54 | 0.69 |

Values will vary by model, prompt, decoding settings, and precision.

## Runtime Constraints & Best Practices

1. **Attention Implementation**: Use eager attention (`attn_implementation="eager"`) if flash attention drops hidden state hooks.
2. **Numeric Stability**: Spectral decomposition (eigenvalue calculation) is sensitive to precision. We recommend running covariance eigendecomposition on CPU or using `float32` tensors on CUDA to avoid early-step NaNs.
3. **Memory Management**: Telemetry collection inherently stores layer-wise activations. Scale your batch sizes down compared to standard generation workloads.

## Validated Research Findings

PRISM telemetry has been deployed in large-scale causal intervention sweeps to study model collapse, semantic grounding, and continuous evaluation protocols.

### Key Mechanistic Attributes Discovered:
* **Compression Bottlenecks**: Analysis reliably demonstrates that self-attention actively compresses representational rank, while SwiGLU / FFN blocks restore dimension capacity (Rank Restoration Profiles).
* **Early-to-Mid Causal Corridors**: Semantic steering and causal interventions (like Activation Patching) show highest efficacy boundaries operating predominantly in early-to-mid layers (e.g., L7-L11 bounds in 32-layer architectures).
* **Temporal Paradox boundary**: Model token interventions show an irreversible plateau (e.g., around 24 tokens) where exogenous correction fails if applied after the generator begins a collapse loop.

*(Note: Specific quantitative outcomes vary by model lineage. The above represents structural phenomena observed and validated during local test sweeps.)*

## Testing

PRISM ships a comprehensive `pytest` suite (**165 tests** as of March 2026) covering all analytical and evaluation components:

| Module group | Test file(s) |
|---|---|
| Core telemetry schemas | `test_telemetry.py` |
| Report schema & adapter coverage | `test_adapter_coverage.py`, `test_telemetry.py` |
| Attention circuits | `test_attention.py` |
| Causal patching | `test_causal.py` |
| Geometry / Hessian | `test_integration_advanced_geometry.py`, `test_hessian_diagnostics.py` |
| Logit / Tuned lens | `test_lens.py` |
| MLP decomposition | `test_mlp.py` |
| Probing & steering | `test_probing.py` |
| SAE features | `test_sae.py` |
| Evaluation metrics | `test_eval_metrics.py`, `test_eval_calibration.py` |
| Drift & early warning | `test_eval_early_warning.py`, `test_eval_drift_metrics.py` |
| Temporal collapse | `test_eval_temporal.py` |
| Hybrid / arch | `test_hybrid.py` |
| Integration / master scan | `test_integration_master_scan.py`, `test_integration_advanced_diagnostics.py` |
| Math proofs | `test_math_*.py` |

Run the full suite:

```bash
pip install -e ".[dev]"
pytest
```

## Model Run Logs

For bounded model analyses and comparison sweeps, use `scripts/run_model_analysis.py`.
It writes a canonical Prism log bundle under `logs/model-runs/` and mirrors the same
run into the model-local `analysis/prism/model-runs/` folder when the target repo is local.
Each bundle records the run summary, findings, lessons, comparison basis, phase logs,
and mirrored artifacts so the same output can later be rendered in the Console or on the website.

## Future Model Intake

Use PRISM as the shared analysis lane for new models before you promote them
into a larger local or cloud workflow.

Recommended order:

1. Register the model in a short intake note.
2. Run a bounded Prism scan or shared test batch.
3. Write the run and analysis into `MODEL_RUN_TEMPLATE.md`.
4. Compare against the prior model and record what changed.

See `FUTURE_MODEL_INTAKE.md` for the reusable checklist.

## Contributing

We welcome contributions to expand PRISM's telemetry capabilities, add support for new architectures, or introduce new analytical metrics. Please keep all new modules covered by tests in `tests/`.

See `CONTRIBUTING.md` for issue and pull-request guidance.

## Citation

If you use this repository in academic work, cite it as software:

```bibtex
@software{prism_2026,
  title = {PRISM: Phase-based Research & Interpretability Spectral Microscope},
  author = {HumanAI Convention Contributors},
  year = {2026},
  url = {https://github.com/humanaiconvention/prism}
}
```

## License

This project is licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
