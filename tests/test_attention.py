import pytest
import torch
import torch.nn as nn
from prism.attention.circuits import AttentionAnalyzer

class DummyModel(nn.Module):
    def __init__(self, hidden_size=32, num_heads=4, num_kv_heads=2):
        super().__init__()
        self.config = type('Config', (), {
            'hidden_size': hidden_size,
            'num_attention_heads': num_heads,
            'num_key_value_heads': num_kv_heads,
            'num_hidden_layers': 2
        })()
        
        self.device = "cpu"
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([self._make_layer(hidden_size) for _ in range(2)])
        
    def _make_layer(self, hidden_size):
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        layer.self_attn.k_proj = nn.Linear(hidden_size, hidden_size//2, bias=False)
        layer.self_attn.v_proj = nn.Linear(hidden_size, hidden_size//2, bias=False)
        layer.self_attn.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        return layer
        
    def forward(self, input_ids, output_attentions=False):
        # Fake forward pass that just calls hooks if registered
        for layer in self.model.layers:
            # Fake attention weights: (batch, heads, seq, seq)
            attn_weights = torch.rand(input_ids.shape[0], self.config.num_attention_heads, input_ids.shape[1], input_ids.shape[1])
            layer.self_attn(input_ids, out=(None, attn_weights))

def test_analyze_head_ov():
    model = DummyModel()
    analyzer = AttentionAnalyzer(model)
    
    # 1. Test OV analysis
    metrics = analyzer.analyze_head_ov(layer_idx=0, head_idx=0)
    
    assert "effective_rank" in metrics
    assert "utilization" in metrics
    
    assert isinstance(metrics["effective_rank"], float)
    assert isinstance(metrics["utilization"], float)
    
    assert metrics["effective_rank"] >= 1.0

def test_detect_induction_heads():
    # Because our dummy model just returns random attention weights,
    # we just want to ensure it runs without crashing and returns the dict.
    model = DummyModel()
    # Add a mock to self_attn to accept forward calls
    def attach_mock(layer_instance):
        def mock_forward(*args, **kwargs):
            if 'out' in kwargs:
                # Trigger hooks manually
                out = kwargs['out']
                for hook in layer_instance.self_attn._forward_hooks.values():
                    hook(layer_instance.self_attn, args, out)
            return args[0]
        layer_instance.self_attn.forward = mock_forward
        
    for layer in model.model.layers:
        attach_mock(layer)

    analyzer = AttentionAnalyzer(model)
    
    class DummyTokenizer:
        pass
        
    scores = analyzer.detect_induction_heads(DummyTokenizer(), pattern_length=10, repeat_count=2)
    
    assert len(scores) == 2 # 2 layers
    for layer_idx, head_scores in scores.items():
        assert len(head_scores) == 4 # 4 heads
        assert all(isinstance(s, float) for s in head_scores)
