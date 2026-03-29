"""Architecture adapters for locating model-specific transformer components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn


def _resolve_attr_chain(root: Any, path: Sequence[str]) -> Any | None:
    current = root
    for name in path:
        if not hasattr(current, name):
            return None
        current = getattr(current, name)
    return current


def _is_layer_container(value: Any) -> bool:
    if isinstance(value, (list, tuple, nn.ModuleList)):
        return len(value) > 0
    return (
        hasattr(value, "__len__")
        and hasattr(value, "__getitem__")
        and not isinstance(value, (str, bytes, dict, torch.Tensor))
    )


@dataclass(frozen=True)
class ResolvedLayerLayout:
    path: Tuple[str, ...]
    layers: Tuple[nn.Module, ...]


class TransformerArchitectureAdapter:
    """Locates common decoder-only transformer pieces without hard-coding one layout."""

    LAYER_PATHS: Tuple[Tuple[str, ...], ...] = (
        ("model", "layers"),
        ("model", "decoder", "layers"),
        ("model", "blocks"),
        ("model", "decoder", "blocks"),
        ("decoder", "layers"),
        ("decoder", "blocks"),
        ("encoder", "block"),
        ("decoder", "block"),
        ("transformer", "blocks"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("layers",),
        ("h",),
        ("blocks",),
        ("encoder", "layer"),
        ("encoder", "layers"),
    )
    FINAL_NORM_PATHS: Tuple[Tuple[str, ...], ...] = (
        ("model", "norm"),
        ("model", "final_layer_norm"),
        ("model", "final_norm"),
        ("model", "ln_final"),
        ("model", "norm_final"),
        ("model", "decoder", "norm"),
        ("model", "decoder", "layer_norm"),
        ("model", "decoder", "final_norm"),
        ("model", "decoder", "ln_final"),
        ("model", "decoder", "norm_final"),
        ("decoder", "norm"),
        ("decoder", "final_layer_norm"),
        ("decoder", "layer_norm"),
        ("decoder", "final_norm"),
        ("decoder", "ln_final"),
        ("decoder", "norm_final"),
        ("encoder", "layer_norm"),
        ("encoder", "final_layer_norm"),
        ("gpt_neox", "final_layer_norm"),
        ("transformer", "ln_f"),
        ("transformer", "norm_f"),
        ("transformer", "norm"),
        ("transformer", "final_norm"),
        ("transformer", "ln_final"),
        ("transformer", "norm_final"),
        ("norm",),
        ("final_layer_norm",),
        ("final_norm",),
        ("ln_f",),
        ("ln_final",),
        ("norm_final",),
        ("norm_out",),
        ("layer_norm",),
    )
    ATTENTION_NAMES: Tuple[str, ...] = (
        "self_attn",
        "self_attention",
        "SelfAttention",
        "EncDecAttention",
        "attn",
        "attention",
    )
    FUSED_QKV_NAMES: Tuple[str, ...] = ("query_key_value", "qkv", "Wqkv", "c_attn", "in_proj", "qkv_proj")
    ATTENTION_OUTPUT_NAMES: Tuple[str, ...] = ("o_proj", "Wo", "dense", "out_proj", "c_proj", "o")
    MLP_NAMES: Tuple[str, ...] = ("mlp", "feed_forward", "FeedForward", "ffn", "DenseReluDense", "block_sparse_moe", "moe")
    MLP_OUTPUT_NAMES: Tuple[str, ...] = ("down_proj", "fc2", "wo", "dense_4h_to_h", "w2", "out_proj", "c_proj")
    LM_HEAD_NAMES: Tuple[str, ...] = (
        "lm_head",
        "embed_out",
        "output_projection",
        "output_embeddings",
        "output_layer",
        "output_head",
        "unembed",
        "unembedding",
    )

    def __init__(self, model: Any, strict: bool = False):
        self.model = model
        self._layout = self._resolve_layer_layout(strict=strict)
        self._final_norm = self._find_final_norm() or nn.Identity()
        self._lm_head = self._find_lm_head()

    @property
    def layers(self) -> Tuple[nn.Module, ...]:
        return self._layout.layers

    @property
    def layer_path(self) -> Tuple[str, ...]:
        return self._layout.path

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def describe(self) -> dict[str, Any]:
        config = getattr(self.model, "config", None)
        return {
            "layer_path": ".".join(self.layer_path) if self.layer_path else "",
            "num_layers": self.num_layers,
            "hidden_size": self._safe_int(
                getattr(config, "hidden_size", getattr(config, "n_embd", getattr(config, "d_model", 0))),
                0,
            ),
            "num_hidden_layers": self._safe_int(getattr(config, "num_hidden_layers", self.num_layers), self.num_layers),
            "attention_names": self.ATTENTION_NAMES,
            "mlp_names": self.MLP_NAMES,
        }

    def resolve_layer(self, layer_idx: int) -> nn.Module:
        if not self.layers:
            raise ValueError("No transformer layers were resolved for this model.")
        return self.layers[layer_idx]

    def hidden_state_index_for_layer(self, layer_idx: int) -> int:
        return layer_idx + 1

    def resolve_attention_module(self, layer_or_idx: int | nn.Module) -> nn.Module:
        layer = self.resolve_layer(layer_or_idx) if isinstance(layer_or_idx, int) else layer_or_idx
        module = self._resolve_first_attr_recursive(self.ATTENTION_NAMES, layer)
        if module is None:
            if self._has_any_direct_attr(
                layer,
                (*self.ATTENTION_NAMES, *self.FUSED_QKV_NAMES, "q_proj", "k_proj", "v_proj", "o_proj", "Wq", "Wk", "Wv", "Wo", "q", "k", "v", "o"),
            ):
                return layer
            raise ValueError(f"Could not locate an attention module on layer type {type(layer).__name__}.")
        return module

    def resolve_mlp_module(self, layer_or_idx: int | nn.Module) -> nn.Module:
        layer = self.resolve_layer(layer_or_idx) if isinstance(layer_or_idx, int) else layer_or_idx
        module = self._resolve_first_attr_recursive(self.MLP_NAMES, layer)
        if module is None:
            if self._has_any_direct_attr(
                layer,
                ("fc1", "fc2", "wi", "wo", "up_proj", "down_proj", "dense_h_to_4h", "dense_4h_to_h", "w1", "w2"),
            ):
                return layer
            raise ValueError(f"Could not locate an MLP module on layer type {type(layer).__name__}.")
        return module

    def resolve_final_norm(self) -> nn.Module | None:
        return self._final_norm

    def resolve_lm_head(self) -> nn.Module | None:
        return self._lm_head

    def resolve_attention_projections(self, layer_or_idx: int | nn.Module) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        attn = self.resolve_attention_module(layer_or_idx)
        for names in (("q_proj", "k_proj", "v_proj", "o_proj"), ("Wq", "Wk", "Wv", "Wo"), ("q", "k", "v", "o")):
            if all(hasattr(attn, name) for name in names):
                return tuple(getattr(attn, name).weight.detach() for name in names)  # type: ignore[return-value]
        fused = self._resolve_first_attr_recursive(self.FUSED_QKV_NAMES, attn)
        if fused is not None:
            q_proj, k_proj, v_proj = self._split_fused_qkv(self._projection_weight(fused))
            out_proj = self._resolve_first_attr_recursive(self.ATTENTION_OUTPUT_NAMES, attn)
            if out_proj is None:
                raise ValueError(
                    f"Attention module {type(attn).__name__} exposes fused QKV weights but no output projection."
                )
            return q_proj, k_proj, v_proj, self._projection_weight(out_proj)
        raise ValueError(
            f"Attention module {type(attn).__name__} does not expose standard q/k/v/o projection weights."
        )

    def resolve_mlp_down_projection(self, layer_or_idx: int | nn.Module) -> torch.Tensor:
        mlp = self.resolve_mlp_module(layer_or_idx)
        down_proj = self._resolve_first_attr_recursive(self.MLP_OUTPUT_NAMES, mlp)
        if down_proj is not None:
            return self._projection_weight(down_proj)
        raise ValueError(f"MLP module {type(mlp).__name__} does not expose a recognized output projection.")

    def _resolve_layer_layout(self, strict: bool = False) -> ResolvedLayerLayout:
        for root in self._candidate_roots():
            for path in self.LAYER_PATHS:
                container = _resolve_attr_chain(root, path)
                if _is_layer_container(container):
                    layers = tuple(container)
                    if layers:
                        return ResolvedLayerLayout(path=path, layers=layers)
        if strict:
            raise ValueError(
                "Could not locate transformer layers. Supported layouts include: "
                + ", ".join(".".join(path) for path in self.LAYER_PATHS)
            )
        return ResolvedLayerLayout(path=(), layers=())

    def _find_final_norm(self) -> nn.Module | None:
        for root in self._candidate_roots():
            for path in self.FINAL_NORM_PATHS:
                value = _resolve_attr_chain(root, path)
                if isinstance(value, nn.Module):
                    return value
        return None

    def _find_lm_head(self) -> nn.Module | None:
        for root in self._candidate_roots():
            value = self._resolve_first_attr_recursive(self.LM_HEAD_NAMES, root)
            if value is not None:
                return value
            getter = getattr(root, "get_output_embeddings", None)
            if callable(getter):
                value = getter()
                if isinstance(value, nn.Module):
                    return value
        return None

    def _candidate_roots(self) -> Tuple[Any, ...]:
        roots = [self.model]
        base_model = getattr(self.model, "base_model", None)
        if isinstance(base_model, nn.Module) and base_model is not self.model:
            roots.append(base_model)
        return tuple(roots)

    @staticmethod
    def _resolve_first_attr(names: Iterable[str], root: Any) -> Any | None:
        for name in names:
            if hasattr(root, name):
                return getattr(root, name)
        return None

    @staticmethod
    def _resolve_first_attr_recursive(names: Iterable[str], root: Any) -> Any | None:
        value = TransformerArchitectureAdapter._resolve_first_attr(names, root)
        if value is not None:
            return value
        if isinstance(root, nn.Module):
            for module in root.modules():
                if module is root:
                    continue
                value = TransformerArchitectureAdapter._resolve_first_attr(names, module)
                if value is not None:
                    return value
        return None

    @staticmethod
    def _has_any_direct_attr(root: Any, names: Iterable[str]) -> bool:
        return any(hasattr(root, name) for name in names)

    @staticmethod
    def _projection_weight(candidate: Any) -> torch.Tensor:
        if isinstance(candidate, torch.Tensor):
            return candidate.detach()
        if hasattr(candidate, "weight"):
            weight = getattr(candidate, "weight")
            if isinstance(weight, torch.Tensor):
                return weight.detach()
        if hasattr(candidate, "W") and isinstance(getattr(candidate, "W"), torch.Tensor):
            return getattr(candidate, "W").detach()
        raise ValueError(f"Cannot extract projection weights from {type(candidate).__name__}.")

    @staticmethod
    def _split_fused_qkv(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if weight.ndim != 2:
            raise ValueError("Fused QKV weights must be a 2D matrix.")
        if weight.shape[0] % 3 == 0:
            q_proj, k_proj, v_proj = torch.chunk(weight.detach(), 3, dim=0)
            return q_proj, k_proj, v_proj
        if weight.shape[1] % 3 == 0:
            q_proj, k_proj, v_proj = torch.chunk(weight.detach(), 3, dim=1)
            return q_proj.t().contiguous(), k_proj.t().contiguous(), v_proj.t().contiguous()
        raise ValueError(
            f"Unable to split fused QKV weight with shape {tuple(weight.shape)} into three equal parts."
        )

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        try:
            if value is None:
                return default
            return int(value)
        except (TypeError, ValueError):
            return default


# Backward-compatible alias for callers that prefer the shorter name.
ArchitectureAdapter = TransformerArchitectureAdapter
