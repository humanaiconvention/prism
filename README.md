# Spectral Microscope

**A causal inference toolkit for analyzing internal hidden state trajectories and attention routing dynamics in hybrid and pure transformer architectures.**

Companion code to the papers *"The Post-Hoc Illusion"* and *"Hybrid Attractor Formation via Temporal Cross-Component Resonance"*.

## Overview
The Spectral Microscope is a PyTorch-based forward-hook telemetry system designed for token-by-token autoregressive generation capture. It correctly records *inline* attention routing and spectral geometries (effective dimension, spectral entropy), avoiding the causal mismatch that plagues post-hoc replay analysis.

## Features
- **Inline Spectral Analysis:** Captures streaming covariance and effective intrinsic dimension dynamics without interrupting generated context loops.
- **Autoregressive Hooks:** Records true self-attention mappings at generation time.
- **Reproducibility:** Completely separated from proprietary environment strings.

## Installation

```bash
git clone https://github.com/humanaiconvention/spectral-microscope.git
cd spectral-microscope
pip install -r requirements.txt
```

### Critical Hardware and Framework Caveats

> [!WARNING]
> Please adhere to the following framework and hardware requirements, as ignoring them violates the geometric invariants of the analysis:

1. **Eager Attention is Mandatory:** You *must* pass `attn_implementation="eager"` when loading HuggingFace models. Flash Attention and SDPA fuse the attention components and do not explicitly materialize the raw attention matrices, returning `None` to the hooks.
2. **Qwen Model Precision Bug:** When evaluating Qwen2.5 series models on CUDA, standard `float16` produces NaN logits from step 0. You must use `--dtype bfloat16` (or `.to(torch.bfloat16)`) for all Qwen testbed models.
3. **Spectral Rank Operations (Replay Mode):** If running spectral covariance decompositions (e.g. `eigvalsh`) on certain hardware, the calculation must be executed on the CPU due to memory bounds and CUDA precision quirks. The analyzer implementation handles this by enforcing `.float().cpu()` before the covariance step.

## Usage

### Analyzing Generation Trajectories
You can run an inline telemetry pass using the wrapper:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.spectral_microscope import SpectralMicroscope

model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto",
    attn_implementation="eager" # Critical
)

microscope = SpectralMicroscope(max_tokens=64)
result = microscope.generate_and_analyze(
    model=model,
    tokenizer=tokenizer,
    prompt="Explain the theory of general relativity."
)

print(result["response"])
for step in result["telemetry"]:
    print(f"Step {step['step']} | Entropy: {step['spectral_entropy']:.2f}")
```

### Reproducing Paper Figures
To regenerate the core empirical figures without requiring a GPU, run:

```bash
python reproduce_paper.py --data_dir data_release --output_dir figures
```

## Citation
If you use this tool in your research, please cite the corresponding paper:
*"The Post-Hoc Illusion: Why Replay-Based Attention Analysis Fails in a Hybrid Convolutional-Attention Language Model."* (2026)
