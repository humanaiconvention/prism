"""``prism.provenance`` — Model lineage detection via Cisco's MPK.

PRISM's geometry tools describe *what a model looks like*; PRISM's NLA
integration describes *what a layer is thinking*; this submodule answers
*where did this model come from?* by wrapping Cisco's Model Provenance
Kit (Apache-2.0, https://github.com/cisco-ai-defense/model-provenance-kit).

Surface
-------
::

    from prism.provenance import (
        scan_model_provenance,        # database lookup
        compare_models,                # pairwise comparison
        ProvenanceResult,              # typed result
        ProvenanceMatch,
        ProvenanceSignals,
        MPKBackend,                    # real backend (requires provenancekit pkg)
        mock_scan, mock_compare,       # offline / test backend
        DEFAULT_PROVENANCE_THRESHOLD,  # 0.70
    )

Caveats (verbatim from the MPK README)
--------------------------------------
* MPK output is **strong evidence … but not absolute proof**.  Statistical
  fingerprints, not cryptographic signatures.
* MPK cannot tell whether weights were copied or trained independently
  when architectures are identical.
* MPK is brand new — v1.0.0 was released 2026-05-04.  Treat its
  ground-truth claims with the deference that maturity earns.

Every :class:`ProvenanceResult` from this submodule carries
``not_cryptographic=True`` so audit consumers cannot accidentally drop
the caveat when serialising results.
"""

from __future__ import annotations

from .client import MPKBackend, compare_models, scan_model_provenance
from .mock import mock_compare, mock_scan
from .types import (
    DEFAULT_PROVENANCE_THRESHOLD,
    ProvenanceMatch,
    ProvenanceResult,
    ProvenanceSignals,
)

__all__ = [
    "DEFAULT_PROVENANCE_THRESHOLD",
    "ProvenanceResult",
    "ProvenanceMatch",
    "ProvenanceSignals",
    "MPKBackend",
    "scan_model_provenance",
    "compare_models",
    "mock_scan",
    "mock_compare",
]
