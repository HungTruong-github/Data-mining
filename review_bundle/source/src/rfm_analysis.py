"""
Module rfm_analysis: Functional alias for RFM feature engineering.

Core implementation is maintained in src/feature_engineering.py.
This module re-exports RFM feature extraction and segmentation utilities
for backward compatibility and clean semantic imports.
"""
from src.feature_engineering import (
    build_rfm_features,
    create_rfm_scores,
    assign_rfm_segment,
    summarize_rfm_segments,
    summarize_features,
)

__all__ = [
    "build_rfm_features",
    "create_rfm_scores",
    "assign_rfm_segment",
    "summarize_rfm_segments",
    "summarize_features",
]
