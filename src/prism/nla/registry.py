"""Registry of pre-trained Natural Language Autoencoder (NLA) checkpoints.

Anthropic released NLA training code and checkpoints in May 2026 (see
https://transformer-circuits.pub/2026/nla/index.html). The kitft/nla-models
collection on the Hugging Face Hub hosts the released artifacts.

This module is a thin lookup table: it records which target models have
a publicly available NLA and where to find it. It does **not** download
weights — that's the inference backend's job.

IMPORTANT: There is no released NLA for Gemma-4-E2B-it (the HumanAI
Convention production family). Running one of the listed NLAs on
Gemma-4-E2B activations would be methodologically invalid because the
AR's affine map is tied to the architecture it was trained against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class NLACheckpoint:
    """Metadata for a released NLA checkpoint."""

    nla_id: str
    """Hugging Face Hub identifier for the NLA artifact."""

    target_model: str
    """Hugging Face Hub identifier of the target LLM the NLA was trained for."""

    target_layer: int
    """Layer index in the target model whose residual stream this NLA verbalizes."""

    total_layers: int
    """Total transformer block count of the target model (for context)."""

    d_model: int
    """Width of the target's residual stream — input dimension for the NLA."""

    license: str = "Apache-2.0"
    """License attached to the released checkpoint."""

    notes: str = ""


# Released by kitft following Anthropic's May 2026 NLA paper. Layer
# selections match the depths Anthropic reports in the paper appendices
# (roughly 2/3 through the network). d_model values come from the
# corresponding target model configs on the Hub.
_REGISTRY: Dict[str, NLACheckpoint] = {
    "kitft/nla-qwen2.5-7b-instruct-layer20": NLACheckpoint(
        nla_id="kitft/nla-qwen2.5-7b-instruct-layer20",
        target_model="Qwen/Qwen2.5-7B-Instruct",
        target_layer=20,
        total_layers=28,
        d_model=3584,
    ),
    "kitft/nla-gemma-3-12b-it-layer32": NLACheckpoint(
        nla_id="kitft/nla-gemma-3-12b-it-layer32",
        target_model="google/gemma-3-12b-it",
        target_layer=32,
        total_layers=48,
        d_model=3840,
    ),
    "kitft/nla-gemma-3-27b-it-layer41": NLACheckpoint(
        nla_id="kitft/nla-gemma-3-27b-it-layer41",
        target_model="google/gemma-3-27b-it",
        target_layer=41,
        total_layers=62,
        d_model=5376,
    ),
    "kitft/nla-llama-3.3-70b-instruct-layer53": NLACheckpoint(
        nla_id="kitft/nla-llama-3.3-70b-instruct-layer53",
        target_model="meta-llama/Llama-3.3-70B-Instruct",
        target_layer=53,
        total_layers=80,
        d_model=8192,
    ),
}


def list_checkpoints() -> List[NLACheckpoint]:
    """Return every registered NLA checkpoint."""
    return list(_REGISTRY.values())


def get_checkpoint(nla_id: str) -> Optional[NLACheckpoint]:
    """Look up a checkpoint by its NLA identifier. Returns ``None`` if unknown."""
    return _REGISTRY.get(nla_id)


def find_for_target(target_model: str) -> List[NLACheckpoint]:
    """Return all registered NLAs trained for *target_model*.

    Useful for the website surface: given a model the user wants to scan,
    show which NLAs are available (or an empty list, meaning none).
    """
    return [c for c in _REGISTRY.values() if c.target_model == target_model]


def is_supported(target_model: str) -> bool:
    """``True`` if a pre-trained NLA exists for *target_model*."""
    return len(find_for_target(target_model)) > 0
