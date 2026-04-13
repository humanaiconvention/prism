from .viability import GeometricViability
from .core import (
    VALID_SPECTRAL_MODES,
    compute_cosine,
    split_vector_by_direction,
    parse_spectral_modes,
    project_noise_to_component,
    project_orthogonal_noise,
    fit_principal_basis,
    unit_vector,
    compute_coeff,
    orthogonal_residual,
    fit_pca_bank,
    make_random_orthogonal_subspace,
    project_state,
    project_onto_basis,
    project_out_basis,
    compute_mean_cosine_to_ref,
    apply_givens_rotations,
    outlier_geometry,
    outlier_geometry_numpy,
)
from .scanner import (
    scan_model_geometry,
    DEFAULT_PROBE_PROMPT,
    HOSTILITY_WARN_THRESHOLD,
)

__all__ = [
    # high-level API
    "scan_model_geometry",
    "DEFAULT_PROBE_PROMPT",
    "HOSTILITY_WARN_THRESHOLD",
    # tensor/array geometry
    "outlier_geometry",
    "outlier_geometry_numpy",
    # geometric analysis
    "GeometricViability",
    # geometry primitives
    "VALID_SPECTRAL_MODES",
    "compute_cosine",
    "split_vector_by_direction",
    "parse_spectral_modes",
    "project_noise_to_component",
    "project_orthogonal_noise",
    "fit_principal_basis",
    "unit_vector",
    "compute_coeff",
    "orthogonal_residual",
    "fit_pca_bank",
    "make_random_orthogonal_subspace",
    "project_state",
    "project_onto_basis",
    "project_out_basis",
    "compute_mean_cosine_to_ref",
    "apply_givens_rotations",
]
