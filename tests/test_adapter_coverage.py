import torch
import torch.nn as nn

from prism.attention.circuits import AttentionAnalyzer
from prism.architecture import TransformerArchitectureAdapter


class FusedAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.query_key_value = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.dense = nn.Linear(hidden_size, hidden_size, bias=False)


class DummyLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.self_attn = FusedAttentionBlock(hidden_size)
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)


class WrappedDummyModel(nn.Module):
    def __init__(self, hidden_size: int = 8, intermediate_size: int = 16, vocab_size: int = 32):
        super().__init__()
        self.config = type(
            "Config",
            (),
            {
                "hidden_size": hidden_size,
                "num_hidden_layers": 1,
                "_name_or_path": "wrapped-dummy",
            },
        )()
        self.base_model = nn.Module()
        self.base_model.layers = nn.ModuleList([DummyLayer(hidden_size, intermediate_size)])
        self.base_model.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)


def test_adapter_resolves_wrapped_fused_attention_and_mlp():
    model = WrappedDummyModel()
    adapter = TransformerArchitectureAdapter(model)

    assert adapter.num_layers == 1
    assert adapter.layer_path == ("layers",)
    assert adapter.resolve_final_norm() is not None
    assert adapter.resolve_lm_head() is model.lm_head

    q_proj, k_proj, v_proj, o_proj = adapter.resolve_attention_projections(0)
    assert q_proj.shape == (8, 8)
    assert k_proj.shape == (8, 8)
    assert v_proj.shape == (8, 8)
    assert o_proj.shape == (8, 8)

    down_proj = adapter.resolve_mlp_down_projection(0)
    assert down_proj.shape == (8, 16)

    description = adapter.describe()
    assert description["num_layers"] == 1
    assert description["hidden_size"] == 8


class T5SelfAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o = nn.Linear(hidden_size, hidden_size, bias=False)


class DenseReluDense(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.wi = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.wo = nn.Linear(intermediate_size, hidden_size, bias=False)


class T5LikeBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.SelfAttention = T5SelfAttention(hidden_size)
        self.DenseReluDense = DenseReluDense(hidden_size, intermediate_size)


class T5LikeModel(nn.Module):
    def __init__(self, hidden_size: int = 8, intermediate_size: int = 16, vocab_size: int = 32):
        super().__init__()
        self.config = type(
            "Config",
            (),
            {
                "hidden_size": hidden_size,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "_name_or_path": "t5-like-dummy",
            },
        )()
        self.decoder = nn.Module()
        self.decoder.block = nn.ModuleList([T5LikeBlock(hidden_size, intermediate_size)])
        self.decoder.final_layer_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)


class OptLikeLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.self_attn = FusedAttentionBlock(hidden_size)
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=False)


class OptLikeModel(nn.Module):
    def __init__(self, hidden_size: int = 8, intermediate_size: int = 16, vocab_size: int = 32):
        super().__init__()
        self.config = type(
            "Config",
            (),
            {
                "hidden_size": hidden_size,
                "num_hidden_layers": 1,
                "_name_or_path": "opt-like-dummy",
            },
        )()
        self.base_model = nn.Module()
        self.base_model.layers = nn.ModuleList([OptLikeLayer(hidden_size, intermediate_size)])
        self.base_model.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)


class FalconLikeBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.self_attention = nn.Module()
        self.self_attention.query_key_value = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.self_attention.dense = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp = nn.Module()
        self.mlp.dense_h_to_4h = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.mlp.dense_4h_to_h = nn.Linear(intermediate_size, hidden_size, bias=False)


class FalconLikeModel(nn.Module):
    def __init__(self, hidden_size: int = 8, intermediate_size: int = 16, vocab_size: int = 32):
        super().__init__()
        self.config = type(
            "Config",
            (),
            {
                "hidden_size": hidden_size,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_kv_heads": 2,
                "_name_or_path": "falcon-like-dummy",
            },
        )()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([FalconLikeBlock(hidden_size, intermediate_size)])
        self.transformer.ln_f = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_output_embeddings(self):
        return self.output_projection


def test_adapter_resolves_t5_style_attention_and_direct_ffn():
    t5_model = T5LikeModel()
    t5_adapter = TransformerArchitectureAdapter(t5_model)

    assert t5_adapter.num_layers == 1
    assert t5_adapter.layer_path == ("decoder", "block")
    assert t5_adapter.resolve_final_norm() is t5_model.decoder.final_layer_norm
    assert t5_adapter.resolve_lm_head() is t5_model.lm_head

    t5_q, t5_k, t5_v, t5_o = t5_adapter.resolve_attention_projections(0)
    assert t5_q.shape == (8, 8)
    assert t5_k.shape == (8, 8)
    assert t5_v.shape == (8, 8)
    assert t5_o.shape == (8, 8)

    t5_down_proj = t5_adapter.resolve_mlp_down_projection(0)
    assert t5_down_proj.shape == (8, 16)

    opt_model = OptLikeModel()
    opt_adapter = TransformerArchitectureAdapter(opt_model)

    assert opt_adapter.num_layers == 1
    assert opt_adapter.layer_path == ("layers",)
    assert opt_adapter.resolve_mlp_module(0) is opt_model.base_model.layers[0]
    opt_down_proj = opt_adapter.resolve_mlp_down_projection(0)
    assert opt_down_proj.shape == (8, 16)


def test_adapter_resolves_falcon_style_blocks_and_output_embedding_fallback():
    falcon_model = FalconLikeModel()
    falcon_adapter = TransformerArchitectureAdapter(falcon_model)

    assert falcon_adapter.num_layers == 1
    assert falcon_adapter.layer_path == ("transformer", "h")
    assert falcon_adapter.resolve_final_norm() is falcon_model.transformer.ln_f
    assert falcon_adapter.resolve_lm_head() is falcon_model.output_projection

    falcon_q, falcon_k, falcon_v, falcon_o = falcon_adapter.resolve_attention_projections(0)
    assert falcon_q.shape == (8, 8)
    assert falcon_k.shape == (8, 8)
    assert falcon_v.shape == (8, 8)
    assert falcon_o.shape == (8, 8)

    falcon_down_proj = falcon_adapter.resolve_mlp_down_projection(0)
    assert falcon_down_proj.shape == (8, 16)

    analyzer = AttentionAnalyzer(falcon_model, adapter=falcon_adapter)
    assert analyzer.n_kv_heads == 2
    assert analyzer.heads_per_group == 2
