"""Validated tensor and provenance adapters for the multiscale WSI pilot."""

from .multiscale_bag import (
    FEATURE_DIM,
    FeatureMatrixError,
    MultiscaleBag,
    build_multiscale_bag,
    concatenate_feature_matrices,
    load_feature_matrix,
    validate_feature_matrix,
)
from .omic import (
    BLCA_PILOT_CASE_ID,
    BLCA_PILOT_DIMS,
    BLCA_PILOT_SLIDE_ID,
    OmicContractError,
    PatientOmicModalities,
    load_blca_pilot_omics,
    load_patient_omic_modalities,
)
from .padding import (
    PaddedWSIBatch,
    pad_patient_bags,
    to_released_healnet_orientation,
    validate_padded_batch,
)
from .provenance import (
    BranchProvenanceSpec,
    PatchProvenance,
    ProvenanceError,
    build_two_scale_provenance,
    provenance_as_dicts,
    validate_provenance_alignment,
    write_provenance_csv,
)

__all__ = [
    "FEATURE_DIM",
    "BLCA_PILOT_CASE_ID",
    "BLCA_PILOT_DIMS",
    "BLCA_PILOT_SLIDE_ID",
    "BranchProvenanceSpec",
    "FeatureMatrixError",
    "MultiscaleBag",
    "OmicContractError",
    "PaddedWSIBatch",
    "PatchProvenance",
    "PatientOmicModalities",
    "ProvenanceError",
    "build_multiscale_bag",
    "build_two_scale_provenance",
    "concatenate_feature_matrices",
    "load_feature_matrix",
    "load_blca_pilot_omics",
    "load_patient_omic_modalities",
    "pad_patient_bags",
    "provenance_as_dicts",
    "to_released_healnet_orientation",
    "validate_feature_matrix",
    "validate_padded_batch",
    "validate_provenance_alignment",
    "write_provenance_csv",
]
