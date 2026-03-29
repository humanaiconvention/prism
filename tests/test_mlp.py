import pytest
import torch
import torch.nn as nn
from prism.mlp.memory import MLPAnalyzer

class DummyModel(nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.config = type('Config', (), {'hidden_size': hidden_size})()
        
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            nn.Module() for _ in range(2)
        ])
        for layer in self.model.layers:
            layer.mlp = nn.Module()
            layer.mlp.gate_proj = nn.Linear(hidden_size, hidden_size * 4)
            layer.mlp.up_proj = nn.Linear(hidden_size, hidden_size * 4)
            layer.mlp.down_proj = nn.Linear(hidden_size * 4, hidden_size)

def test_mlp_analyzer():
    model = DummyModel()
    analyzer = MLPAnalyzer(model)
    
    # Generate some dummy pre/post features
    batch_size = 2
    seq_len = 5
    hidden_size = 16
    
    pre_attention = torch.randn(batch_size, seq_len, hidden_size)
    post_attention = torch.randn(batch_size, seq_len, hidden_size)
    post_mlp = torch.randn(batch_size, seq_len, hidden_size)
    
    # 1. Test Rank Restoration
    profile = analyzer.rank_restoration_profile(pre_attention, post_attention, post_mlp)
    
    assert "input_rank" in profile
    assert "attn_impact" in profile
    assert "mlp_impact" in profile
    assert "net_impact" in profile
        
    # 2. Test Key-Value Mapping extraction
    vocab_size = 100
    lm_head = nn.Linear(hidden_size, vocab_size)
    final_norm = nn.LayerNorm(hidden_size)
    tokenizer = type('Tokenizer', (), {'decode': lambda self, x: str(x)})()
    
    # Extract
    layer_idx = 0
    neuron_idx = 5
    mapped_tokens = analyzer.map_neuron_to_vocabulary(layer_idx, neuron_idx, lm_head, final_norm, tokenizer, k=3)
    
    assert len(mapped_tokens) == 3
    for token, prob in mapped_tokens:
        assert isinstance(token, str)
        assert isinstance(prob, float)
