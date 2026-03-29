"""Phase synchronization and spectral coherence analysis."""

from .coherence import PhaseAnalyzer
from .timing import (
    select_prompt_window_bounds,
    select_generation_window_bounds,
    resolve_composition_window
)
from .synergy import (
    parse_bundle_specs,
    OProjHeadColumnAblation,
    OProjHeadBundleAblation
)

__all__ = [
    "PhaseAnalyzer",
    "select_prompt_window_bounds",
    "select_generation_window_bounds",
    "resolve_composition_window",
    "parse_bundle_specs",
    "OProjHeadColumnAblation",
    "OProjHeadBundleAblation"
]
