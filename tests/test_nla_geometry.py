"""Integration tests for scan_model_geometry's optional NLA pass.

Uses a tiny SmolLM2-135M model on CPU + the deterministic mock NLA so
the suite stays GPU-free and offline-after-cache.
"""

from __future__ import annotations

import pytest
import torch

from prism.geometry import scan_model_geometry
from prism.nla import NLAExplanation, mock_explainer


@pytest.fixture(scope="module")
def smol_model():
    """Smallest model PRISM's existing tests already pull. 135M params, 30 layers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "HuggingFaceTB/SmolLM2-135M"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        device_map="cpu",
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    return model, tok


def test_scan_without_nla_unchanged(smol_model):
    """Back-compat regression: nla_explainer=None must not add an 'nla' key."""
    model, tok = smol_model
    result = scan_model_geometry(model, tokenizer=tok, prompt="The cat sat.")
    assert "nla" not in result
    # Original contract still satisfied
    for key in (
        "model_name",
        "prompt",
        "n_layers",
        "layers",
        "mean_quantization_hostility",
        "worst_layer_idx",
        "best_layer_idx",
        "worst_layer_hostility",
        "best_layer_hostility",
        "n_hostile_layers",
    ):
        assert key in result, f"missing legacy key: {key}"


def test_scan_with_mock_nla_adds_block(smol_model):
    model, tok = smol_model
    d_model = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    target_layer = min(5, n_layers - 1)
    exp = mock_explainer(d_model=d_model, layer_idx=target_layer)

    result = scan_model_geometry(
        model, tokenizer=tok, prompt="The cat sat on the mat.", nla_explainer=exp
    )

    assert "nla" in result
    block = result["nla"]
    assert block["layer_idx"] == target_layer
    assert block["n_samples"] > 0
    assert isinstance(block["summary"], str) and block["summary"]
    assert 0.0 <= block["mean_fve"] <= 1.0
    assert block["fve_std"] >= 0.0
    assert all(isinstance(e, NLAExplanation) for e in block["explanations"])
    assert len(block["explanations"]) == block["n_samples"]


def test_nla_block_does_not_leak_raw_activations(smol_model):
    """The brief is explicit: the nla block must NOT carry raw activation vectors."""
    model, tok = smol_model
    d_model = model.config.hidden_size
    exp = mock_explainer(d_model=d_model, layer_idx=2)
    result = scan_model_geometry(model, tokenizer=tok, nla_explainer=exp)

    block = result["nla"]
    assert "activation_vector" not in block
    assert "activation_vectors" not in block
    # The per-sample explanation may include the AR reconstruction, but
    # never the raw input.
    for e in block["explanations"]:
        assert "activation_vector" not in e.metadata


def test_nla_d_model_mismatch_raises(smol_model):
    model, tok = smol_model
    bad = mock_explainer(d_model=model.config.hidden_size + 7, layer_idx=1)
    with pytest.raises(ValueError, match="d_model"):
        scan_model_geometry(model, tokenizer=tok, nla_explainer=bad)


def test_nla_out_of_range_layer_raises(smol_model):
    model, tok = smol_model
    bad = mock_explainer(
        d_model=model.config.hidden_size,
        layer_idx=model.config.num_hidden_layers + 99,
    )
    with pytest.raises(ValueError, match="outside"):
        scan_model_geometry(model, tokenizer=tok, nla_explainer=bad)


def test_nla_n_samples_capped_to_seq_len(smol_model):
    model, tok = smol_model
    d_model = model.config.hidden_size
    exp = mock_explainer(d_model=d_model, layer_idx=0)
    # Request more samples than the prompt has tokens — scanner must clamp.
    result = scan_model_geometry(
        model, tokenizer=tok, prompt="A B C.", nla_explainer=exp, nla_n_samples=999
    )
    seq_len = len(tok("A B C.")["input_ids"])
    assert result["nla"]["n_samples"] == seq_len
