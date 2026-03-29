import torch
import numpy as np
from prism.phase.coherence import PhaseAnalyzer
from prism.entropy.lens import EntropyDynamics
from prism.geometry.viability import GeometricViability

def test_phase_synchronization():
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type('Config', (), {'hidden_size': 512, 'num_hidden_layers': 12})()
    
    model = MockModel()
    phase_analyzer = PhaseAnalyzer(model)
    
    # Create synthetic hidden states (batch, seq, dim)
    t = torch.linspace(0, 2*np.pi, 32)
    hidden_states = torch.sin(t).view(1, 32, 1).repeat(1, 1, 512) + torch.randn(1, 32, 512) * 0.1
    
    phases = phase_analyzer.extract_hilbert_phase(hidden_states)
    assert phases.shape[1] == 32 # check sequence length
    
    plv = phase_analyzer.compute_plv(phases, phases)
    assert isinstance(plv, float)
    
    clust_index = phase_analyzer.phase_clustering_index(phases)
    assert isinstance(clust_index, float)
    
    fft_res = phase_analyzer.fft_telemetry(hidden_states)
    assert 'dominant_freq' in fft_res

def test_entropy_reduction_dynamics():
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type('Config', (), {'hidden_size': 512, 'num_hidden_layers': 12})()
    
    model = MockModel()
    entropy_dyn = EntropyDynamics(model)
    
    probs = torch.tensor([0.8, 0.1, 0.05, 0.05])
    h_05 = entropy_dyn.compute_renyi_entropy(probs, alpha=0.5)
    assert isinstance(h_05, float)
    
    layer_entropies = [4.5, 4.8, 4.2, 3.9, 3.5]
    profile = entropy_dyn.entropy_profile_tracking(layer_entropies)
    assert isinstance(profile, list)
    assert len(profile) == len(layer_entropies) - 1
    
    autocorr = entropy_dyn.entropy_autocorrelation(layer_entropies)
    assert isinstance(autocorr, float)

def test_viability_constraints():
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type('Config', (), {'hidden_size': 512, 'num_hidden_layers': 12})()
    
    model = MockModel()
    geom_viability = GeometricViability(model)
    
    v_score = geom_viability.compute_viability_score(effective_dimension=185.0, hidden_dim=576)
    assert isinstance(v_score, float)
