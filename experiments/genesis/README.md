# Mechanistic Interpretability of Genesis-152M (PRISM Replication)

**Model**: [guiferrarib/genesis-152m-instruct](https://huggingface.co/guiferrarib/genesis-152m-instruct)  
**Version**: Phase 12F (Circuit Closure & Interface Characterisation)
**Framework**: PRISM (Phase-based Research & Interpretability Spectral Microscope)

---

## Objective

Perform mechanistic interpretability research on **Genesis-152M-Instruct**, a 152M-parameter hybrid linear+softmax attention model. The goals are to:
1. Characterize the spectral geometry of the residual stream during autoregressive generation.
2. Understand how the two mixer types (GLA linear attention and FoX forgetting attention) contribute to representation structure.
3. Investigate the causes and consequences of observed representational compression.
4. Map causal roles of individual attention heads in the FoX layers.
5. Identify the causal mechanisms of semantic steering and the fragility of OOD transfer.

### Local-First & Windows-Native Research
A primary goal of this replication suite is to demonstrate that high-quality mechanistic interpretability can be performed on local Windows hardware. All experiments in this suite were conducted natively on Windows 11 using a consumer NVIDIA GPU, achieving ~7.3 tokens/sec via optimized PyTorch fallbacks—bypassing the need for Linux-only Triton kernels or complex WSL2 setups.

---

## Architecture

| Property | Value |
|---|---|
| **Parameters** | 151.76M |
| **Layers** | 30 (23 GLA + 7 FoX) |
| **Hidden dim (n_embd)** | 576 |
| **Intermediate size** | 1,440 (~2.5x hidden) |
| **Attention heads** | 9 query / 3 KV (GQA 3:1, head_dim=64) |
| **GLA layers (23)** | Gated DeltaNet — O(n) linear attention with delta rule recurrence |
| **FoX layers (7)** | Forgetting Transformer — O(n²) softmax attention with learned forget gate |
| **TTT** | Metacognition (rank=4, inner LR 0.01, dual mode) |
| **Selective Activation** | Top-k sparsity (k=85%) on SwiGLU FFN |
| **Normalization** | ZeroCenteredRMSNorm throughout |
| **Position encoding** | Partial RoPE (50% rotation) in GLA; NoPE in FoX |
| **µP** | Maximal Update Parametrization (base_width=256) |
| **Context length** | 2,048 |
| **Package** | `genesis-llm` v2.0.3 |

### Layer Layout (verified at runtime)

FoX layers at every 4th position: `(layer_idx + 1) % 4 == 0`

```
FoX (softmax + forget gate):  [3, 7, 11, 15, 19, 23, 27]     (7 layers)
GLA (DeltaNet linear):        [0,1,2, 4,5,6, 8,9,10, 12,13,14,
                                16,17,18, 20,21,22, 24,25,26, 28,29] (23 layers)
```

---

## Experimental Results

### Finding 1: Effective Rank — Corrected Measurement
True output ER at the final layer is **185.5/576 (32.2%)** at N=28,224. Prior measurements (8.3%) were artifacts of short sequence lengths (T=64). ER plateaus in L23-L27 (~31-34%) before dropping at L29.

### Finding 2: The Mixer Is the Compression Bottleneck
Sub-block ER analysis (N=3840) reveals a massive ~4x per-block oscillation:
- **ZeroCenteredRMSNorm**: Inflates rank (+154 ER @ L15).
- **Mixer (GLA/FoX)**: The real bottleneck, crushing rank (-136 ER @ L15).
- **SwiGLU FFN**: Major rank restorer (+124 ER @ L15).
- **Residual Add**: Re-aligns but does not act as an independent bottleneck.

### Finding 3: Per-Layer Sub-Block Profile
Norm inflation is depth-dependent (U-shaped CKA profile, 0.73 at L16). Mixer compression varies from 26% (L3) to 68% (L27).

### Finding 4: TTT Contribution is Negligible
TTT metacognition (rank=4) structurally contributes **< 0.5 ER variance** to the global subspace. Its influence is a local manifold nudge, not macroscopic expansion.

### Finding 5: Aligned Block Injection
Block contributions are small (ratio 0.1-0.3) and positively aligned (cos +0.1 to +0.24) with the residual stream. Blocks nudge the stream rather than overwriting it.

### Finding 6: Prompt-Type Retro-Retraction
While initial runs suggested Math > Creative volume, bootstrapping (Finding 18) showed the gap is not statistically significant (p=0.929). Geometric separation (Principal Angles, Finding 12) remains the valid differentiator.

### Finding 7: FoX Head Specialization
Ablation of **L15-H3** causes mixer ER to skyrocket (+56.2), proving it is a massive rank compressor. Other heads act as load-bearing processors (deltas -10 to -50).

### Finding 8: Two-Stage Attention Compression
Compression happens in:
1. **Head-Internal**: Heads themselves lose variance in mid/deep layers.
2. **W_o Anisotropic Funneling**: Output projection W_o crushes 70-100 dims to align with dominant residual directions.

### Finding 9: Prompt Diversity > Sequence Depth
Statistical power constant (N=3840), prompt distribution diversity injects considerably more orthogonal variance (+13.5 ER) than autoregressive depth.

### Finding 10: GLA Fixed-Point Attractor
GLA recurrent memory subspace converges to a stable geometric orientation (locked at 46.2° relative to T=0) at step **t=27**. Long-context ER (Finding 19) stabilizes at ~13 dimensions.

### Finding 11: Task-Vector Causal Steering
Steering creative generation with a 'Math' task vector (delta_perp) at λ=5.0 successfully shifts output cadence and geometry (+40% closer to Math centroid) without destroying entropy. λ=12.5 causes coherence collapse.

### Finding 12: Shared Syntax vs Orthogonal Semantics
Principal Angle analysis shows a **Universal Trunk** (30.8% of dims < 30° apart) and **Orthogonal Branches** (10.8% of dims > 80° apart) between Math and Creative domains.

### Finding 13: The Lexical Crossover (L15)
At the L15 geometric bottleneck, vocabulary ER collapses from >100 to ~28 words. L15 is the causal mechanism that finalizes sequence prediction.

### Finding 14: Causal Semantic Bottleneck
Orthogonal noise injection at L15 degrades perplexity ($p=1.04 \times 10^{-4}$), proving the model uses the quiet subspace for semantic routing.

### Finding 15: Rotational Dynamics
The residual stream is an oscillatory linear dynamical system. 97.2% of representational modes are complex conjugate pairs, indicating rotation through state space across depth.

### Finding 16: Period-4 Architectural Oscillation
Welford ER tracking reveals periodic volume shifts: GLA layers build representation, FoX layers reconcile/crush them.

### Finding 17: Residual Stream FFT
Fourier spectrum of inter-layer deltas exhibits a power spike at $f=0.25$ (period-4), matching the (GLA/GLA/GLA/FoX) architecture.

### Finding 18: Semantic Volume Insensitivity
Cluster bootstrapping (1,000 iterations) proves spectral volume is insensitive to prompt complexity. Differences are in orientation, not volume.

### Finding 19: GLA Recurrent Compression
GLA state compresses to a highly efficient ~13-dimensional manifold at T=1024, confirming spectral norm stability.

---

## The Causal Reliability Program (Phase 9-12)

### Phase 9: The Patching Paradox
Steering (adding a vector) remains the best causal evidence. Activation Patching (overwriting) remains negative/diagnostic-only on the 24-pair shared benchmark. Results indicate the model uses 'Aligned Injection' rather than simple state replacement.

### Phase 10: Mediator Corridor (L7-L11)
- **10A/J**: Same-vector steering is strongest in a FoX 'mediator corridor' spanning L7 and L11 (+0.0284 margin).
- **10C/K**: OOD transfer is **family-sensitive** and fragile. The corridor is real in-domain but fails to generalize robustly.
- **10D**: Residual decomposition favors a **manifold-nudge operator** account over a static feature-axis story.
- **10E/F**: Localizes the effect to the attention-output corridor with transformed downstream carry.
- **10L/M**: Donor swaps suggest state-conditional structure, especially at L7 with low-overlap donors.
- **10O-R**: Supports an in-domain corridor-input subspace geometry, but necessity remains non-robust and sufficiency is family-specific.
- **10S**: The corridor is a **depth-band** effect, not uniquely FoX-specific.
- **10W-Y**: Rejects portable native scalar-value reuse and coordinated bundle transplants.

### Phase 11: Orthogonal-Remainder Decomposition
Decomposing the natural interchange hint into basis components failed to clear strict all-items promotion gates. No portable component mechanism found.

### Phase 12: Circuit Closure & Interface Characterization
- **12B**: The corridor is sharply **answer-adjacent** (t-1), but effects are not semantic-specific vs random controls.
- **12D**: Local forget-gate control is effective but behaviorally insufficient.
- **12E**: L11 W_o is a **high-gain output stage**, but its gain is non-selective.
- **12F**: Sparse upstream head-bundles do not provide semantic-specific necessity.
**Verdict**: The corridor is a weak, non-specific answer-adjacent access/gain interface.

---

## Status Summary

| Phase | Status | Conclusion |
|---|---|---|
| Spectral Geometry | ✅ Complete | Genesis is compressed (32.2% ER); Mixer/W_o is the bottleneck. |
| Interventions | ✅ Complete | L15 is a causal bottleneck; steering works via manifold-nudge. |
| Mediator Band | ✅ Complete | L7-L11 is the most responsive steering corridor. |
| OOD Robustness | ✅ Complete | Transfer is family-sensitive and fragile. |
| Circuit Closure | ✅ Complete | L11 W_o is a high-gain interface, not a specific semantic circuit. |

---

## Usage (Replication Orchestrator)

The PRISM Phase-based Research orchestrator allows reproducing any finding:

```bash
# List all phases
python go.py --list

# Get info on a phase
python go.py --info 10A

# Run specific phase
python go.py 10A
```

---

## References
*See GENESIS_SPECTRAL_README.md for full citations.*
