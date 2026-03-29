import pytest
import torch
import torch.nn as nn
from prism.probing.linear import LinearProbe, ConceptProber, SteeringVectorExtractor

def test_linear_probe():
    hidden_size = 16
    probe = LinearProbe(hidden_size)
    
    # 1. Forward
    inputs = torch.randn(5, hidden_size)
    outputs = probe(inputs)
    assert outputs.shape == (5, 1)  # Linear outputs (N, 1) usually
    
    # 2. Get direction
    direction = probe.net[0].weight.data.detach().clone().squeeze()
    assert direction.shape == (hidden_size,)

def test_concept_prober():
    hidden_size = 16
    prober = ConceptProber(hidden_size=hidden_size)
    
    features = torch.randn(10, hidden_size)
    labels = torch.randint(0, 2, (10,)).float()
    
    layer_idx = 0
    # Simulate a trained probe
    acc = prober.train_layer_probe(layer_idx, features, labels, epochs=2)
    assert isinstance(acc, float)
    
    assert layer_idx in prober.probes
    assert isinstance(prober.probes[layer_idx], LinearProbe)
    
    # Evaluate extraction
    direction = prober.get_concept_direction(layer_idx)
    assert direction.shape == (hidden_size,)
    
    cka = prober.compute_cka(features, features)
    assert 0.99 <= cka <= 1.01

def test_steering_vector_extractor():
    extractor = SteeringVectorExtractor()
    clean = torch.randn(5, 16)
    corrupt = torch.randn(5, 16)
    
    vec = extractor.extract_cip_vector(clean, corrupt)
    assert vec.shape == (16,)
