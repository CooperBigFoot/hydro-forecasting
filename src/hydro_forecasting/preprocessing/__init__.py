"""Hydro forecasting preprocessing module."""

# Import transformers to trigger registration
from . import log_scale, normalize, standard_scale

# Import main classes
from .base import HydroTransformer, register_transformer
from .grouped import GroupedPipeline
from .log_scale import LogTransformer
from .normalize import NormalizeTransformer
from .pipeline_builder import PipelineBuilder, PipelineSection

# Import individual transformers
from .standard_scale import StandardScaleTransformer
from .transformer_registry import create_sklearn_pipeline, get_transformer_class, list_available_transformers
from .unified import UnifiedPipeline

__all__ = [
    # Main classes
    "HydroTransformer",
    "GroupedPipeline",
    "UnifiedPipeline",
    "PipelineBuilder",
    "PipelineSection",
    # Registry functions
    "register_transformer",
    "get_transformer_class",
    "list_available_transformers",
    "create_sklearn_pipeline",
    # Individual transformers
    "StandardScaleTransformer",
    "NormalizeTransformer",
    "LogTransformer",
]
