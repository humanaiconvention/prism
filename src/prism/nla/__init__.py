"""``prism.nla`` — Natural Language Autoencoder integration.

NLAs (Anthropic, May 2026 — https://transformer-circuits.pub/2026/nla/index.html)
turn raw transformer activations into human-readable explanations.  PRISM
measures activation **geometry** (structure); NLAs measure activation
**semantics** (content).  They're complementary, and this submodule provides
the wiring so :func:`prism.geometry.scan_model_geometry` can attach NLA text
to the per-layer geometry block.

Surface
-------
::

    from prism.nla import (
        NLAExplainer,        # HTTP client for a remote AR server
        NLAExplanation,      # single-sample result
        NLABatchResult,      # aggregate over a layer's samples
        NLACheckpoint,       # registry entry for a pre-trained NLA
        mock_explainer,      # deterministic mock for tests
        list_checkpoints,
        get_checkpoint,
        find_for_target,
        is_supported,
        summarize_layer,     # used by scan_model_geometry
    )

Disclaimers (verbatim from the Anthropic paper)
-----------------------------------------------

1. **Confabulation** — NLA explanations can contain claims about the
   target model's input context that are verifiably false.
2. **Blackbox by construction** — NLAs are not mechanistic; they're a
   learned decoder.
3. **Excessive expressivity** — AV is a full language model and can make
   extra inferences beyond what's in the activation.
4. **Cost** — joint RL on two LMs.

There is no released NLA for Gemma-4-E2B-it. Do not run an NLA trained
for one architecture against a different architecture's activations.
"""

from __future__ import annotations

from .inference import NLAExplainer
from .mock import MockNLAExplainer, mock_explainer
from .registry import (
    NLACheckpoint,
    find_for_target,
    get_checkpoint,
    is_supported,
    list_checkpoints,
)
from .summary import summarize_layer
from .types import NLABatchResult, NLAExplanation

__all__ = [
    # Core types
    "NLAExplanation",
    "NLABatchResult",
    "NLACheckpoint",
    # Inference
    "NLAExplainer",
    "MockNLAExplainer",
    "mock_explainer",
    # Registry helpers
    "list_checkpoints",
    "get_checkpoint",
    "find_for_target",
    "is_supported",
    # Aggregation
    "summarize_layer",
]
