# Natural Language Autoencoders in PRISM

> Integration design doc for the `prism.nla` submodule introduced in
> PRISM 0.3.0.  Audience: contributors and users who want to attach
> human-readable activation explanations to PRISM's geometry scan.

---

## 1. The technique

A **Natural Language Autoencoder (NLA)** converts a transformer's
internal activations into a natural-language description of what the
layer appears to be "thinking" at that position.  Anthropic introduced
the technique in May 2026:

- Blog post: <https://www.anthropic.com/research/natural-language-autoencoders>
- Paper: <https://transformer-circuits.pub/2026/nla/index.html>
- Reference implementation (Apache-2.0):
  <https://github.com/kitft/natural_language_autoencoders>

### Two-model autoencoder

| Component | What it is | What it does |
|---|---|---|
| **AV** — Activation Verbalizer | The target LLM with a learned input head | Receives a single activation token; generates an explanation autoregressively (temperature 1). |
| **AR** — Activation Reconstructor | First *ℓ* layers of the target LLM + a learned affine map | Reads the AV's explanation back; reconstructs the residual stream at layer *ℓ*. |

The pair is trained jointly with

```
L = E[ || h_ℓ − AR(AV(h_ℓ)) ||² ]   +   β · KL(AV_φ ‖ AV_φ_init)
```

— MSE on the layer-*ℓ* residual stream, plus a KL term that keeps the
verbalizer from drifting away from fluent English.

### What it tells you

For a single activation vector *h_ℓ*, the NLA gives you:

1. **Text** — the AV's natural-language description.
2. **Reconstruction FVE** — fraction of variance explained by the AR's
   reconstruction.  Anthropic reports 0.6–0.8 on Claude Haiku 3.5/4.5
   and Opus 4.6.  Low FVE means the verbal description is missing
   structure that's actually in the activation, so the text is less
   trustworthy.

PRISM's `geometry` metrics describe *structure* of hidden states
(outlier ratio, kurtosis, axis alignment).  NLAs describe *semantics*.
They're complementary — running both on the same layer gives you
"what does this layer look like, and what is it thinking about?" in one
pass.

---

## 2. What's released

Anthropic and the kitft repo have published checkpoints for four
target models.  PRISM ships them as registry entries
(`prism/nla/registry.py`):

| `nla_id` | Target model | Layer | Layers (total) | `d_model` |
|---|---|---|---|---|
| `kitft/nla-qwen2.5-7b-instruct-layer20` | Qwen/Qwen2.5-7B-Instruct | 20 | 28 | 3584 |
| `kitft/nla-gemma-3-12b-it-layer32` | google/gemma-3-12b-it | 32 | 48 | 3840 |
| `kitft/nla-gemma-3-27b-it-layer41` | google/gemma-3-27b-it | 41 | 62 | 5376 |
| `kitft/nla-llama-3.3-70b-instruct-layer53` | meta-llama/Llama-3.3-70B-Instruct | 53 | 80 | 8192 |

All four are released under **Apache-2.0**.  PRISM itself stays under
**CC BY 4.0** — only the inference wire format is described here, not
any code from the kitft repository.

---

## 3. What's NOT available

> **There is no released NLA for `google/gemma-4-e2b-it`.**

This is the production family for the HumanAI Convention's Gemma4Good
work.  Training a new NLA from scratch is expensive (SFT on 2×H100‑80GB
followed by RL on 2×8×H100), so we have not trained one and PRISM does
not pretend that it has.

**Do not** run a Gemma-3 NLA against Gemma-4 activations.  The AR
contains a learned affine map tied to a specific architecture's
residual-stream geometry; feeding it activations from a different model
produces output that looks like an explanation but is methodologically
invalid.  `prism.geometry.scan_model_geometry` enforces this: passing a
mismatched `d_model` raises `ValueError`.

---

## 4. Honest disclosed limitations

These are reproduced from Anthropic's paper.  Carry them into any
public-facing artefact that uses NLA output:

1. **Confabulation.**  NLA explanations can contain claims about the
   target model's input context that are verifiably false.
2. **Blackbox by construction.**  NLAs are not mechanistic; they're a
   learned decoder.
3. **Excessive expressivity.**  AV is a full language model and can
   make extra inferences beyond what's actually in the activation.
4. **Cost.**  Joint RL on two language models.

In short: treat NLA text as a *hypothesis generator*, not as ground
truth about what a layer is doing.

---

## 5. PRISM integration

### Inference contract

The released NLAs are designed to be served behind an HTTP endpoint
(SGLang in the kitft reference).  The server receives one or more
activation vectors via `input_embeds` (the activation is injected
directly into the residual stream — no tokenisation step) and returns
the AV's generated text plus an FVE score.

`prism.nla.NLAExplainer` is a thin client over that contract:

```python
from prism.nla import NLAExplainer

exp = NLAExplainer.from_pretrained(
    "kitft/nla-gemma-3-12b-it-layer32",
    server_url="http://my-sglang-host:8000/nla",
)
result = exp.explain(activation_vector)   # 1-D np.ndarray, len = d_model
# result.text                : str   — AV's generated explanation
# result.reconstruction_fve  : float — per-sample FVE, in [0, 1]
# result.reconstructed_vector: np.ndarray — AR's output (same shape)
# result.metadata            : dict  — model_id, layer_idx, backend, ...

results = exp.explain_batch(list_of_activations)
```

PRISM never holds NLA weights locally — they stay on the model server.
Only the activation vector and the generated text traverse the wire.

### Pluggable transport

For testing (or for in-process inference) you can inject a transport:

```python
def fake_transport(payload: dict) -> dict:
    return {"text": "fake explanation", "reconstruction_fve": 0.7}

exp = NLAExplainer(checkpoint, transport=fake_transport)
```

The transport may be a callable (`payload -> response`) or an object
exposing `post(url, json)`.  The default — used when only `server_url=`
is supplied — wraps `requests` lazily, so PRISM does not gain a hard
dependency on it.

### Hooking into `scan_model_geometry`

```python
from prism.geometry import scan_model_geometry
from prism.nla import NLAExplainer

exp = NLAExplainer.from_pretrained(
    "kitft/nla-gemma-3-12b-it-layer32",
    server_url="http://localhost:8000/nla",
)
result = scan_model_geometry(
    "google/gemma-3-12b-it",
    nla_explainer=exp,
)

# result["nla"] = {
#     "layer_idx":    32,
#     "n_samples":    16,
#     "explanations": [<NLAExplanation>, ...],
#     "summary":      "common themes (5): agreement, syntactic, ...",
#     "mean_fve":     0.71,
#     "fve_std":      0.04,
# }
```

If `nla_explainer=None` (the default), the result dict is byte-identical
to PRISM's prior `scan_model_geometry` output.  Existing callers — the
`humanaiconvention-prism` PyPI consumers and `D:/gemma4good/prism_integration/`
in particular — are unaffected.

The `nla` block intentionally does **not** echo raw activation vectors
back to the caller.  The whole point of NLAs is to surface text;
re-emitting the input would defeat that.

---

## 6. When to use NLA vs geometry alone

| Question | Tool |
|---|---|
| "Will this layer survive 4-bit quantisation?" | `outlier_geometry` / `scan_model_geometry` |
| "How did my fine-tune reshape the residual manifold?" | `scan_model_geometry` across checkpoints |
| "What is layer 32 thinking about on this prompt?" | NLA via `scan_model_geometry(..., nla_explainer=...)` |
| "Is the model attending to PII?" | NLA, with the **confabulation caveat** above |

NLA output is most useful when paired with a geometry signal that
suggests something interesting is happening at a layer (a hostility
spike, a kurtosis outlier, an unexpected rank).  Reading the NLA text
in isolation — across hundreds of positions — is rarely actionable.

---

## 7. Cost transparency

Training an NLA from scratch requires roughly:

- **Stage 1 (SFT):** ≈2 × H100‑80GB.
- **Stage 2 (RL):** ≈2 × 8 × H100 (joint training on AV + AR).

PRISM does not train NLAs.  Using a released NLA in PRISM only needs
the AR/AV model to be serving on a GPU host; the PRISM-side client is
pure CPU/numpy.

---

## 8. Talking to kitft's raw SGLang endpoint

The default `NLAExplainer` expects a **wrapper API**: a single endpoint
that takes `{activation_vector, nla_id, layer_idx}` and returns
`{text, reconstruction_fve, reconstructed_vector?}`.  This is **not**
the wire format kitft's released `nla_inference.py` uses.

kitft uses two phases:

1. POST the activation to SGLang's `/generate` as
   `{input_embeds: <numpy_bytes>, sampling_params: {temperature: 1.0, max_new_tokens: 200, ...}}`.
   Response: `{text: "..."}` — text only.
2. Run the AR (first ℓ layers of the target + an affine map) against
   the AV's output to compute the reconstruction and the FVE.  This is
   typically done in-process by `nla_inference.py` or against a second
   server.

If you have a real kitft setup, write a `transport=` callable that
bridges the two formats:

```python
import requests
from prism.nla import NLAExplainer, get_checkpoint

def kitft_transport(sglang_url, ar_runner):
    """Closure: turns PRISM's wrapper payload into kitft's two-phase flow."""
    def _t(payload):
        # Phase 1 — verbalize
        resp = requests.post(
            f"{sglang_url}/generate",
            json={
                "input_embeds": payload["activation_vector"],
                "sampling_params": {"temperature": 1.0, "max_new_tokens": 200},
            },
            timeout=120,
        )
        resp.raise_for_status()
        text = resp.json()["text"]
        # Phase 2 — score (ar_runner is your local AR inference function)
        fve, rec = ar_runner(payload["activation_vector"], text)
        return {"text": text, "reconstruction_fve": fve, "reconstructed_vector": rec}
    return _t

exp = NLAExplainer(
    get_checkpoint("kitft/nla-gemma-3-12b-it-layer32"),
    transport=kitft_transport("http://my-sglang:30000", my_local_ar_runner),
)
```

This is intentionally a **two-step bridge** rather than baked into
`prism.nla` — PRISM does not pretend to know your AR serving topology.
A future `KitftSGLangBackend` class could be added once a reference
deployment exists to test against; until then, the transport callable
is the canonical extension point.

Status (2026-05-11): no end-to-end test exists in PRISM that talks to
a real kitft server.  HAIC's local BEAST (RTX 2080, 8 GB VRAM) cannot
host a 12 B AR, so live validation is deferred until a remote NLA host
is available.

---

## 9. Versioning

NLA support landed in PRISM **1.1.0**.  The integration is
additive and back-compatible — every call site that worked in 0.2.0
continues to work in 0.3.0 with no change.

See `CHANGELOG.md` for the change list, and `tests/test_nla_inference.py`
+ `tests/test_nla_geometry.py` for the executable specification.
