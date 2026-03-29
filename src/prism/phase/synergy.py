import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Any
from ..architecture import TransformerArchitectureAdapter

def _normalize_bundle_spec(layer_heads: List[Tuple[int, Tuple[int, ...]]]) -> str:
    return ";".join(f"{layer}:{'|'.join(str(head) for head in heads)}" for layer, heads in layer_heads)

def parse_bundle_specs(raw_bundle_specs: str) -> List[Dict[str, Any]]:
    """Parses bundle specifications grouping specific heads across layers.
    Format: name=layer:head|head[;layer:head|head]
    E.g. "l7_top3=7:5|6|0,l11_top3=11:5|1|0,joint_l7_l11_top6=7:5|6|0;11:5|1|0"
    """
    bundles = []
    seen_names = set()
    for token in raw_bundle_specs.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Invalid bundle spec '{token}'. Expected name=layer:head|head[;layer:head|head].")
        name, raw_spec = token.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid bundle spec '{token}': missing bundle name.")
        if name in seen_names:
            raise ValueError(f"Duplicate bundle name: {name}")
            
        layer_heads = []
        seen_pairs = set()
        for segment in raw_spec.split(";"):
            segment = segment.strip()
            if not segment:
                continue
            if ":" not in segment:
                raise ValueError(f"Invalid bundle segment '{segment}' in '{token}'.")
            raw_layer, raw_heads = segment.split(":", 1)
            layer = int(raw_layer.strip())
            heads = []
            for raw_head in raw_heads.split("|"):
                raw_head = raw_head.strip()
                if not raw_head:
                    continue
                head = int(raw_head)
                pair = (layer, head)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                heads.append(head)
            if not heads:
                raise ValueError(f"Bundle '{name}' has an empty head list for layer {layer}.")
            layer_heads.append((layer, tuple(heads)))
            
        if not layer_heads:
            raise ValueError(f"Bundle '{name}' is empty.")
        seen_names.add(name)
        bundles.append(
            {
                "bundle_name": name,
                "bundle_spec": _normalize_bundle_spec(layer_heads),
                "layer_heads": tuple(layer_heads),
                "bundle_size": int(sum(len(heads) for _, heads in layer_heads)),
            }
        )
    if not bundles:
        raise ValueError("No bundle specs parsed.")
    return bundles


class OProjHeadColumnAblation:
    """Ablates a specific attention head's column in the output projection matrix."""
    def __init__(self, model: nn.Module, ablation_layer: int, ablation_head: int, head_dim: int):
        self.model = model
        self.adapter = TransformerArchitectureAdapter(model)
        self.ablation_layer = int(ablation_layer)
        self.ablation_head = int(ablation_head)
        self.head_dim = int(head_dim)
        self.saved_slice = None

    def _resolve_output_projection(self) -> torch.Tensor:
        layer = self.adapter.resolve_layer(self.ablation_layer)
        attn = self.adapter.resolve_attention_module(layer)
        o_proj = getattr(attn, "o_proj", None)
        if o_proj is None or not hasattr(o_proj, "weight"):
            raise ValueError(f"Layer {self.ablation_layer} has no recognized attention output projection to ablate")
        return o_proj.weight

    def __enter__(self):
        weight = self._resolve_output_projection()
        start = self.ablation_head * self.head_dim
        end = start + self.head_dim
        
        if end > weight.shape[1]:
            raise ValueError(
                f"Head slice [{start}:{end}] exceeds o_proj input width {weight.shape[1]} at layer {self.ablation_layer}."
            )
            
        with torch.no_grad():
            self.saved_slice = weight[:, start:end].detach().clone()
            weight[:, start:end].zero_()
        return self

    def __exit__(self, exc_type, exc, tb):
        weight = self._resolve_output_projection()
        start = self.ablation_head * self.head_dim
        end = start + self.head_dim
        with torch.no_grad():
            weight[:, start:end].copy_(self.saved_slice)
        self.saved_slice = None

class OProjHeadBundleAblation:
    """Ablates a bundle of attention heads across multiple layers in the o_proj weights."""
    def __init__(self, model: nn.Module, layer_heads: Tuple[Tuple[int, Tuple[int, ...]], ...], head_dim: int):
        self.model = model
        self.adapter = TransformerArchitectureAdapter(model)
        self.layer_heads = tuple(layer_heads)
        self.head_dim = int(head_dim)
        self.saved_slices = []

    def _resolve_output_projection(self, layer: int) -> torch.Tensor:
        resolved_layer = self.adapter.resolve_layer(layer)
        attn = self.adapter.resolve_attention_module(resolved_layer)
        o_proj = getattr(attn, "o_proj", None)
        if o_proj is None or not hasattr(o_proj, "weight"):
            raise ValueError(f"Layer {layer} has no recognized attention output projection to ablate")
        return o_proj.weight

    def __enter__(self):
        for layer, heads in self.layer_heads:
            weight = self._resolve_output_projection(int(layer))
            for head in heads:
                start = int(head) * self.head_dim
                end = start + self.head_dim
                
                if end > weight.shape[1]:
                    raise ValueError(
                        f"Head slice [{start}:{end}] exceeds o_proj input width {weight.shape[1]} at layer {layer}."
                    )
                with torch.no_grad():
                    saved = weight[:, start:end].detach().clone()
                    weight[:, start:end].zero_()
                self.saved_slices.append((int(layer), int(head), saved))
        return self

    def __exit__(self, exc_type, exc, tb):
        for layer, head, saved in reversed(self.saved_slices):
            weight = self._resolve_output_projection(layer)
            start = head * self.head_dim
            end = start + self.head_dim
            with torch.no_grad():
                weight[:, start:end].copy_(saved)
        self.saved_slices = []
