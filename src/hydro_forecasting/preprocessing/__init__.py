"""Hydro forecasting preprocessing module."""

# Import transformers to trigger registration
from . import standard_scale, normalize, log_scale

# Import main classes
from .base import HydroTransformer, register_transformer
from .grouped import GroupedPipeline
from .unified import UnifiedPipeline
from .pipeline_builder import PipelineBuilder, PipelineSection
from .transformer_registry import (
    get_transformer_class,
    list_available_transformers,
    create_sklearn_pipeline
)

# Import individual transformers
from .standard_scale import StandardScaleTransformer
from .normalize import NormalizeTransformer
from .log_scale import LogTransformer

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