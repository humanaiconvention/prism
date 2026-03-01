# Spectral Microscope

Inline spectral telemetry for autoregressive language-model generation.

## Public Repository Scope

This public repository currently ships:

- `src/spectral_microscope/`: inline telemetry capture library.
- `quickstart.ipynb`: first-run validation notebook.
- `requirements.txt`: dependency pins for the public quickstart path.

This repo does **not** include the full private training/evaluation pipeline for differentiable gated LoRA experiments.

## Public Release Housekeeping

- `AGENTS.md` is intentionally included as maintainer guidance for public-safe updates.
- `reproduce_paper.py` is intentionally not shipped in this release because the prior version was a template with synthetic data.
- Package metadata now lives in `pyproject.toml` for standard `pip install -e .` workflows.

Supporting maintenance documents:

- `CONTRIBUTING.md` for issue and pull-request guidance.
- `CHANGELOG.md` for versioned change history.
- `RELEASE_TODO.md` for the 14-item release checklist status.

## Research Status Snapshot (Synced to Phase 4.0, 2026-02-28)

### Phase 3.0 (Local Proof-of-Concept Run)

- Config: 45 training pairs, 20 epochs.
- Easy prompts: `3.7604 -> 3.6497` (Delta NLL `-0.1107`, `11/12` improved).
- Hard prompts: `6.1458 -> 5.9062` (Delta NLL `-0.2396`, `12/12` improved).
- Gate discrimination (hard - easy): `0.3543`.

### Phase 3.0 (Scaled Run)

- Config: 3,591 contrastive pairs, rank 32, 30 epochs.
- Easy Delta NLL: `-0.610`.
- Medium (held-out) Delta NLL: `-0.577`.
- Hard Delta NLL: `-0.764`.
- Gate means: Easy `0.129`, Hard `0.915`, Discrimination `+0.786`.

### Phase 4.0 (Temporal Paradox Boundary)

- Interpretability probe: `corr(gate_mean, baseline_nll) = +0.782`, `p=1.44e-42`.
- Gate bimodality:
  - EASY: 89% low gate (<0.2)
  - HARD: 100% high gate (>=0.2)
  - Leakage: 0/70 easy with gate >= 0.5
  - Misses: 0/57 hard with gate < 0.7

Checkpoint ablation learning curve:

| Epoch | Easy Delta NLL | Hard Delta NLL | Selectivity (Hard-Easy) |
|---|---:|---:|---:|
| ep4  | -0.613 | -0.751 | -0.138 |
| ep10 | -0.612 | -0.764 | -0.152 |
| ep20 | -0.614 | -0.773 | -0.158 |
| ep30 | -0.610 | -0.764 | -0.154 |

Temporal windows on hard prompts:

| Gate Window | Mean Delta NLL | % of Full Benefit |
|---|---:|---:|
| 0->4 tokens  | -0.645 | 84.5% |
| 0->8 tokens  | -0.710 | 93.0% |
| 0->12 tokens | -0.748 | 97.9% |
| 0->16 tokens | -0.762 | 99.8% |
| 0->24 tokens | -0.764 | 100.0% |

Boundary finding: intervention benefit plateaus at token 24.

### Active Experiment Queue

- Rank ablation is currently running for ranks `8/16/32/64`.
- Output target: `logs/rank_ablation_rank{8,16,32,64}_stratified.csv`.

Interim snapshot (3/4 ranks complete; means over 200 stratified prompts):

| Rank | Easy Delta NLL | Hard Delta NLL | Selectivity (Hard-Easy) | Gate Discrimination |
|---:|---:|---:|---:|---:|
| 8  | -0.5790 | -0.7552 | -0.1762 | +0.6927 |
| 16 | -0.6368 | -0.7790 | -0.1422 | +0.6939 |
| 32 | -0.6161 | -0.7794 | -0.1633 | +0.6982 |
| 64 | pending | pending | pending | pending |

Current reading: hard-stratum improvement remains strong across completed ranks, while the final rank-64 run is still in progress.

This section will be finalized once rank-64 evaluation finishes.

## Installation

```bash
git clone https://github.com/humanaiconvention/spectral-microscope.git
cd spectral-microscope
pip install -e .
```

### Notebook Environment (optional)

```bash
pip install -e ".[notebook]"
# or: pip install -r requirements.txt
```

## Usage

### Notebook Path (Recommended)

Open and run:

- `quickstart.ipynb`

### Python API

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from spectral_microscope import SpectralMicroscope, __version__

print(f"spectral_microscope version: {__version__}")

model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    attn_implementation="eager",
)

microscope = SpectralMicroscope(max_tokens=64, window_size=32, streaming_cov_alpha=0.95)
result = microscope.generate_and_analyze(
    model=model,
    tokenizer=tokenizer,
    prompt="Explain what a black hole is in three simple sentences.",
    max_new_tokens=50,
    temperature=0.7,
)

print(result["response"])
print(result["telemetry"][0])
```

### Example Telemetry Output

Representative rows from `pd.DataFrame(result["telemetry"])`:

| step | token | spectral_entropy | effective_dim | streaming_eff_dim | projection_angle |
|---:|---|---:|---:|---:|---:|
| 1 | `The` | 2.08 | 7.42 | 1.00 | 0.74 |
| 2 | ` event` | 2.21 | 8.03 | 1.88 | 0.71 |
| 3 | ` horizon` | 2.33 | 8.67 | 2.54 | 0.69 |
| 4 | ` is` | 2.46 | 9.12 | 3.05 | 0.66 |
| 5 | ` the` | 2.51 | 9.40 | 3.41 | 0.64 |

Values will vary by model, prompt, decoding settings, and precision.

## Runtime Constraints

1. Use eager attention: `attn_implementation="eager"`.
2. Qwen on CUDA: prefer `dtype=torch.bfloat16` to avoid early-step NaN logits.
3. Spectral decomposition: run covariance eigendecomposition on CPU float for stability.

## Telemetry Schema

Per generated token, telemetry includes:

- `spectral_entropy`
- `effective_dim`
- `streaming_eff_dim`
- `projection_angle`

## Citation

If you use this repository in academic work, cite it as software:

```bibtex
@software{spectral_microscope_2026,
  title = {Spectral Microscope: Inline Spectral Telemetry for Autoregressive Language Models},
  author = {HumanAI Convention Contributors},
  year = {2026},
  url = {https://github.com/humanaiconvention/spectral-microscope}
}
```

## License

This project is licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
