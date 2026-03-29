import pytest
import torch
import torch.nn as nn
from prism.causal.patching import ActivationPatcher

class DummyModel(nn.Module):
    def __init__(self, hidden_size=16, num_layers=2):
        super().__init__()
        self.config = type('Config', (), {'num_hidden_layers': num_layers, 'hidden_size': hidden_size})()
        
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])
    
    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return x

def test_activation_patcher():
    model = DummyModel()
    patcher = ActivationPatcher(model)
    
    layer_to_patch = model.model.layers[0]
    cache_key = "layer_0_out"
    
    # 1. Clean run (cache activation)
    clean_input = torch.randn(1, 5, 16)
    
    with patcher.trace():
        patcher.cache_activation(layer_to_patch, cache_key)
        out_clean = model(clean_input)
        
    assert cache_key in patcher._cache
    assert patcher._cache[cache_key].shape == (1, 5, 16)
    
    # 2. Corrupt run (inject activation at position 3)
    corrupt_input = torch.randn(1, 5, 16)
    pos_idx = 3
    
    with patcher.trace():
        patcher.inject_activation(layer_to_patch, cache_key, position_idx=pos_idx)
        out_patched = model(corrupt_input)
        
    # The patched output should differ from a purely corrupt output
    with torch.no_grad():
        out_corrupt = model(corrupt_input)
        
    assert not torch.allclose(out_patched, out_corrupt)

