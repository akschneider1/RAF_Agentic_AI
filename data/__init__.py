"""Data module for state licensing comparison data."""

from .state_licensing_data import (
    LicenseCategory,
    LicenseRequirements,
    COSMETOLOGY_DATA,
    CONTRACTOR_DATA,
    REAL_ESTATE_DATA,
    SAMPLE_STATUTES,
    get_all_data_for_license,
    get_state_data,
    compare_states,
    get_best_practice_state,
    get_model_provisions,
)

__all__ = [
    "LicenseCategory",
    "LicenseRequirements",
    "COSMETOLOGY_DATA",
    "CONTRACTOR_DATA",
    "REAL_ESTATE_DATA",
    "SAMPLE_STATUTES",
    "get_all_data_for_license",
    "get_state_data",
    "compare_states",
    "get_best_practice_state",
    "get_model_provisions",
]
