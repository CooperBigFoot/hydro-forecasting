"""Transformer registry module for managing and accessing registered transformers."""

from sklearn.pipeline import Pipeline

from .base import TRANSFORMER_REGISTRY, HydroTransformer


def get_transformer_class(name: str) -> type[HydroTransformer]:
    """Get transformer class by name.

    Args:
        name: Snake_case name of the transformer

    Returns:
        Transformer class

    Raises:
        ValueError: If transformer name is not found in registry
    """
    if name not in TRANSFORMER_REGISTRY:
        available_names = list_available_transformers()
        raise ValueError(f"Transformer '{name}' not found. Available transformers: {available_names}")

    return TRANSFORMER_REGISTRY[name]


def list_available_transformers() -> list[str]:
    """List all registered transformer names.

    Returns:
        Sorted list of available transformer names
    """
    return sorted(TRANSFORMER_REGISTRY.keys())


def create_sklearn_pipeline(transform_names: list[str], columns: list[str] | None = None) -> Pipeline:
    """Create sklearn Pipeline from list of transform names.

    Args:
        transform_names: List of transformer names to chain
        columns: Columns to transform (passed to each transformer)

    Returns:
        sklearn Pipeline with instantiated transformers

    Raises:
        ValueError: If any transformer name is not found in registry
    """
    if not transform_names:
        raise ValueError("At least one transform name must be provided")

    # Validate all transform names exist
    for name in transform_names:
        if name not in TRANSFORMER_REGISTRY:
            available_names = list_available_transformers()
            raise ValueError(f"Transformer '{name}' not found. Available transformers: {available_names}")

    # Create pipeline steps
    steps = []
    for name in transform_names:
        transformer_class = TRANSFORMER_REGISTRY[name]
        transformer_instance = transformer_class(columns=columns)
        steps.append((name, transformer_instance))

    return Pipeline(steps)
