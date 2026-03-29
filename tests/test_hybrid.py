import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace
from prism.arch.hybrid import HybridDiagnostics

class DummyModel(nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.config = type('Config', (), {'hidden_size': hidden_size})()
        
    def forward(self, input_ids, position_ids=None, output_hidden_states=False):
        batch_size, seq_len = input_ids.shape
        # Return fake hidden states
        out = type('Output', (), {})()
        out.hidden_states = [torch.randn(batch_size, seq_len, self.config.hidden_size) for _ in range(3)]
        return out


class Seq2SeqDummyModel(nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.config = type('Config', (), {'hidden_size': hidden_size})()

    def forward(self, input_ids, position_ids=None, output_hidden_states=False, **kwargs):
        batch_size, seq_len = input_ids.shape
        hidden_states = tuple(
            torch.randn(batch_size, seq_len, self.config.hidden_size) for _ in range(3)
        )
        return SimpleNamespace(
            hidden_states=None,
            decoder_hidden_states=hidden_states,
            last_hidden_state=hidden_states[-1],
        )

def test_hybrid_diagnostics():
    model = DummyModel()
    diag = HybridDiagnostics(model)
    
    # 1. Compare attention entropy
    # (batch, heads, seq, seq)
    softmax_attn = torch.rand(2, 4, 10, 10)
    softmax_attn = softmax_attn / softmax_attn.sum(dim=-1, keepdim=True)
    
    linear_state = torch.randn(2, 10, 16)
    
    res = diag.compare_attention_entropy(softmax_attn, linear_state)
    assert "softmax_entropy" in res
    assert "linear_spectral_entropy" in res
    assert isinstance(res["softmax_entropy"], float)
    assert isinstance(res["linear_spectral_entropy"], float)
    
    # 2. Track recurrent attractors
    # The tracked states need to be 2D matrices (batch, dim) or flat vectors for the svd logic
    states = [torch.randn(20, 16) for _ in range(5)]
    attr_res = diag.track_recurrent_attractors(states)
    
    assert "mean_rotation_angle" in attr_res
    assert "final_angle" in attr_res
    assert "is_saturated" in attr_res
    
    # 3. Measure positional sensitivity
    inputs = {"input_ids": torch.randint(0, 100, (2, 10))}
    sens_res = diag.measure_positional_sensitivity(inputs, layer_idx=0)
    
    assert "positional_drift" in sens_res
    assert "rank_collapse_ratio" in sens_res
    assert "is_spatially_rigid" in sens_res
    assert isinstance(sens_res["positional_drift"], float)
    assert isinstance(sens_res["rank_collapse_ratio"], float)


def test_hybrid_diagnostics_uses_decoder_hidden_states():
    model = Seq2SeqDummyModel()
    diag = HybridDiagnostics(model)

    inputs = {"input_ids": torch.randint(0, 100, (2, 10))}
    sens_res = diag.measure_positional_sensitivity(inputs, layer_idx=0)

    assert "positional_drift" in sens_res
    assert "rank_collapse_ratio" in sens_res
    assert "is_spatially_rigid" in sens_res
