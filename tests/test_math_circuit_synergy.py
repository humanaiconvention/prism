import pytest
import torch
import torch.nn as nn

from prism.phase import (
    parse_bundle_specs,
    OProjHeadColumnAblation,
    OProjHeadBundleAblation
)

# --- Pytest Tests ---

class MockOProj:
    def __init__(self, out_features, in_features):
        self.weight = nn.Parameter(torch.ones(out_features, in_features))

class MockAttn:
    def __init__(self, o_proj):
        self.o_proj = o_proj

class MockBlock:
    def __init__(self, o_proj):
        self.attn = MockAttn(o_proj)

class MockModel:
    def __init__(self, num_layers=3, out_features=8, in_features=16):
        self.blocks = [MockBlock(MockOProj(out_features, in_features)) for _ in range(num_layers)]


class AdapterSelfAttention:
    def __init__(self, o_proj):
        self.o_proj = o_proj


class AdapterBlock:
    def __init__(self, o_proj):
        self.SelfAttention = AdapterSelfAttention(o_proj)


class AdapterModel:
    def __init__(self, num_layers=2, out_features=8, in_features=16):
        self.transformer = type("Transformer", (), {})()
        self.transformer.h = [AdapterBlock(AdapterOProj(out_features, in_features)) for _ in range(num_layers)]


class AdapterOProj:
    def __init__(self, out_features, in_features):
        self.weight = nn.Parameter(torch.ones(out_features, in_features))

def test_parse_bundle_specs_single():
    spec = "l7_top3=7:5|6|0"
    bundles = parse_bundle_specs(spec)
    assert len(bundles) == 1
    assert bundles[0]["bundle_name"] == "l7_top3"
    assert bundles[0]["bundle_spec"] == "7:5|6|0"
    assert bundles[0]["layer_heads"] == ((7, (5, 6, 0)),)
    assert bundles[0]["bundle_size"] == 3

def test_parse_bundle_specs_complex():
    spec = "l7_top3=7:5|6|0,l11_top3=11:5|1|0,joint_l7_l11_top6=7:5|6|0;11:5|1|0"
    bundles = parse_bundle_specs(spec)
    assert len(bundles) == 3
    assert bundles[2]["bundle_name"] == "joint_l7_l11_top6"
    assert bundles[2]["layer_heads"] == ((7, (5, 6, 0)), (11, (5, 1, 0)))
    assert bundles[2]["bundle_size"] == 6

def test_parse_bundle_specs_invalid():
    with pytest.raises(ValueError, match="Invalid bundle spec"):
        parse_bundle_specs("bad_spec")
    
    with pytest.raises(ValueError, match="Duplicate bundle name"):
        parse_bundle_specs("dup=7:1,dup=8:2")

def test_oproj_head_column_ablation():
    head_dim = 4
    model = MockModel(num_layers=1, out_features=4, in_features=head_dim*3) # 3 heads
    
    # Target head 1 (columns 4:8)
    with OProjHeadColumnAblation(model, ablation_layer=0, ablation_head=1, head_dim=head_dim):
        weight = model.blocks[0].attn.o_proj.weight
        assert torch.all(weight[:, 0:4] == 1.0) # head 0 untouched
        assert torch.all(weight[:, 4:8] == 0.0) # head 1 ablated
        assert torch.all(weight[:, 8:12] == 1.0) # head 2 untouched

    # Check restoration
    weight = model.blocks[0].attn.o_proj.weight
    assert torch.all(weight == 1.0)

def test_oproj_head_bundle_ablation():
    head_dim = 2
    model = MockModel(num_layers=2, out_features=4, in_features=head_dim*4) # 4 heads per layer
    
    layer_heads = ((0, (1, 3)), (1, (0,)))
    
    with OProjHeadBundleAblation(model, layer_heads, head_dim):
        w0 = model.blocks[0].attn.o_proj.weight
        assert torch.all(w0[:, 0:2] == 1.0) # h0
        assert torch.all(w0[:, 2:4] == 0.0) # h1
        assert torch.all(w0[:, 4:6] == 1.0) # h2
        assert torch.all(w0[:, 6:8] == 0.0) # h3
        
        w1 = model.blocks[1].attn.o_proj.weight
        assert torch.all(w1[:, 0:2] == 0.0) # h0
        assert torch.all(w1[:, 2:8] == 1.0) # h1, h2, h3

    # Check restoration
    assert torch.all(model.blocks[0].attn.o_proj.weight == 1.0)
    assert torch.all(model.blocks[1].attn.o_proj.weight == 1.0)


def test_oproj_head_ablation_adapter_layout():
    head_dim = 4
    model = AdapterModel(num_layers=1, out_features=4, in_features=head_dim * 3)

    with OProjHeadColumnAblation(model, ablation_layer=0, ablation_head=1, head_dim=head_dim):
        weight = model.transformer.h[0].SelfAttention.o_proj.weight
        assert torch.all(weight[:, 0:4] == 1.0)
        assert torch.all(weight[:, 4:8] == 0.0)
        assert torch.all(weight[:, 8:12] == 1.0)

    assert torch.all(model.transformer.h[0].SelfAttention.o_proj.weight == 1.0)
