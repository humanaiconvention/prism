"""High-level geometry scanner: measure quantisation hostility across all layers.

This module provides :func:`scan_model_geometry`, the primary entry point for
researchers who want a quick, model-agnostic quantisation-hostility profile
without any manual hook management.

Typical usage::

    from prism.geometry import scan_model_geometry

    results = scan_model_geometry("google/gemma-4-e2b-it")
    print(results["mean_quantization_hostility"])   # e.g. 0.914 (baseline)

Or with an already-loaded model::

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from prism.geometry import scan_model_geometry

    model = AutoModelForCausalLM.from_pretrained(...)
    tokenizer = AutoTokenizer.from_pretrained(...)
    results = scan_model_geometry(model, tokenizer=tokenizer)
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Union

# ─── optional heavy imports — fail lazily so the pure-NumPy path still works ──

def _require_torch() -> Any:
    try:
        import torch
        return torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for scan_model_geometry().  "
            "Install it with: pip install torch"
        ) from exc


def _require_transformers() -> Any:
    try:
        import transformers
        return transformers
    except ImportError as exc:
        raise ImportError(
            "Hugging Face Transformers is required for scan_model_geometry().  "
            "Install it with: pip install transformers"
        ) from exc


# ─── constants ───────────────────────────────────────────────────────────────

#: Default probe prompt — deliberately varied vocabulary and moderate length so
#: hidden-state statistics are representative rather than degenerate.
DEFAULT_PROBE_PROMPT = (
    "The relationship between cause and effect demonstrates that complex systems "
    "often behave in ways that are difficult to predict from their components alone."
)

#: Threshold above which a layer is considered quantisation-hostile.
HOSTILITY_WARN_THRESHOLD = 0.70


# ─── public API ──────────────────────────────────────────────────────────────

def scan_model_geometry(
    model_or_name: Union[str, Any],
    *,
    tokenizer: Optional[Any] = None,
    prompt: str = DEFAULT_PROBE_PROMPT,
    device: Optional[str] = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    trust_remote_code: bool = False,
) -> Dict[str, Any]:
    """Measure per-layer outlier geometry across every transformer block.

    Attaches lightweight forward hooks to a ``PreTrainedModel``, runs a single
    forward pass with *prompt*, and calls :func:`~prism.geometry.outlier_geometry`
    on the hidden states collected from each layer.

    The function is model-agnostic: it works with any architecture that supports
    ``output_hidden_states=True`` (LLaMA, Gemma, Qwen, Mistral, Phi, GPT-2/NeoX,
    and hundreds of others).  BitsAndBytes 4-bit / 8-bit quantised models are
    handled transparently — activations are cast to ``float32`` before metric
    computation regardless of storage dtype.

    Args:
        model_or_name: Either a :class:`transformers.PreTrainedModel` instance
            (already loaded) **or** a Hugging Face model-id string such as
            ``"google/gemma-4-e2b-it"``.  When a string is passed the model is
            loaded with :func:`transformers.AutoModelForCausalLM.from_pretrained`.
        tokenizer: Tokenizer matching *model_or_name*.  Required when
            *model_or_name* is a :class:`~transformers.PreTrainedModel` instance;
            loaded automatically when *model_or_name* is a string.
        prompt: Text used for the single forward pass.  Longer, more varied
            prompts produce more representative geometry statistics.  Defaults to
            :data:`DEFAULT_PROBE_PROMPT`.
        device: Target device string (e.g. ``"cuda"``, ``"cpu"``).  Defaults to
            ``"cuda"`` if available, else ``"cpu"``.
        load_in_4bit: Load model weights in 4-bit NF4 (requires ``bitsandbytes``
            and ``accelerate``).  Only used when *model_or_name* is a string.
        load_in_8bit: Load model weights in 8-bit LLM.int8.  Only used when
            *model_or_name* is a string.
        trust_remote_code: Passed to ``from_pretrained`` when loading from a
            string identifier.

    Returns:
        A dict with the following keys:

        .. code-block:: text

            model_name                 str    — identifier of the scanned model
            prompt                     str    — probe text used
            n_layers                   int    — number of transformer blocks
            layers                     list   — per-layer result dicts (see below)
            mean_quantization_hostility float  — mean hostility across all layers
            worst_layer_idx            int    — index of most hostile layer
            best_layer_idx             int    — index of least hostile layer
            worst_layer_hostility      float
            best_layer_hostility       float
            n_hostile_layers           int    — layers with hostility > 0.70

        Each element of ``layers`` contains:

        .. code-block:: text

            layer_idx              int
            outlier_ratio          float
            activation_kurtosis    float
            cardinal_proximity     float
            quantization_hostility float

    Raises:
        ImportError: If PyTorch or Transformers is not installed.
        ValueError: If the model does not support ``output_hidden_states=True``.

    Example::

        from prism.geometry import scan_model_geometry

        # 3-line usage: load from HF Hub, profile, inspect
        results = scan_model_geometry("google/gemma-4-e2b-it")
        print(results["mean_quantization_hostility"])
    """
    torch = _require_torch()
    transformers = _require_transformers()

    from .core import outlier_geometry  # local import to avoid circular issues

    # ── resolve device ────────────────────────────────────────────────────────
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── load model + tokenizer if a string name was provided ─────────────────
    if isinstance(model_or_name, str):
        model_name = model_or_name
        _load_kwargs: Dict[str, Any] = {
            "device_map": "auto" if device == "cuda" else device,
            "trust_remote_code": trust_remote_code,
        }
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                _load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            except ImportError as exc:
                raise ImportError(
                    "load_in_4bit=True requires bitsandbytes and accelerate.  "
                    "Install with: pip install 'humanaiconvention-prism[quantized]'"
                ) from exc
        elif load_in_8bit:
            try:
                _load_kwargs["load_in_8bit"] = True
            except Exception as exc:
                raise ImportError(
                    "load_in_8bit=True requires bitsandbytes.  "
                    "Install with: pip install 'humanaiconvention-prism[quantized]'"
                ) from exc

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, **_load_kwargs
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
    else:
        model = model_or_name
        model_name = getattr(model, "name_or_path", type(model).__name__)
        if tokenizer is None:
            raise ValueError(
                "tokenizer must be provided when model_or_name is a PreTrainedModel "
                "instance.  Pass tokenizer=your_tokenizer."
            )

    model.eval()

    # ── tokenise prompt ───────────────────────────────────────────────────────
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    inputs = tokenizer(prompt, return_tensors="pt")

    # Move inputs to the model's first parameter device for non-device_map models
    try:
        first_device = next(model.parameters()).device
        inputs = {k: v.to(first_device) for k, v in inputs.items()}
    except StopIteration:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    # ── forward pass with hidden states ──────────────────────────────────────
    with torch.inference_mode():
        try:
            outputs = model(**inputs, output_hidden_states=True)
        except TypeError:
            raise ValueError(
                f"Model {model_name!r} does not accept output_hidden_states=True.  "
                "PRISM requires a model that exposes intermediate hidden states."
            )

    hidden_states = outputs.hidden_states  # tuple: (embedding, L1, L2, ..., LN)
    if hidden_states is None:
        raise ValueError(
            f"Model {model_name!r} returned None for hidden_states.  "
            "Ensure output_hidden_states=True is supported by this architecture."
        )

    # Skip index 0 (embedding layer before any transformer block)
    layer_hidden_states = hidden_states[1:]

    # ── compute per-layer geometry ────────────────────────────────────────────
    layer_results: List[Dict[str, Any]] = []
    for layer_idx, h in enumerate(layer_hidden_states):
        # h shape: (batch, seq_len, hidden_dim) — squeeze batch dim
        h_2d = h[0].detach()   # (seq_len, hidden_dim), may be float16/bfloat16
        metrics = outlier_geometry(h_2d)
        layer_results.append({"layer_idx": layer_idx, **metrics})

    if not layer_results:
        warnings.warn(
            f"No layer hidden states were collected for {model_name!r}.  "
            "The geometry results will be empty.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {
            "model_name": model_name,
            "prompt": prompt,
            "n_layers": 0,
            "layers": [],
            "mean_quantization_hostility": float("nan"),
            "worst_layer_idx": -1,
            "best_layer_idx": -1,
            "worst_layer_hostility": float("nan"),
            "best_layer_hostility": float("nan"),
            "n_hostile_layers": 0,
        }

    # ── aggregate statistics ──────────────────────────────────────────────────
    hostilities = [r["quantization_hostility"] for r in layer_results]
    mean_h = sum(hostilities) / len(hostilities)
    worst_idx = max(range(len(hostilities)), key=lambda i: hostilities[i])
    best_idx = min(range(len(hostilities)), key=lambda i: hostilities[i])
    n_hostile = sum(1 for h in hostilities if h > HOSTILITY_WARN_THRESHOLD)

    return {
        "model_name": model_name,
        "prompt": prompt,
        "n_layers": len(layer_results),
        "layers": layer_results,
        "mean_quantization_hostility": mean_h,
        "worst_layer_idx": worst_idx,
        "best_layer_idx": best_idx,
        "worst_layer_hostility": hostilities[worst_idx],
        "best_layer_hostility": hostilities[best_idx],
        "n_hostile_layers": n_hostile,
    }
