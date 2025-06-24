"""Tests for transformer registry functionality."""

import pytest
from sklearn.pipeline import Pipeline

from hydro_forecasting.preprocessing import (
    HydroTransformer,
    LogTransformer,
    NormalizeTransformer,
    StandardScaleTransformer,
    create_sklearn_pipeline,
    get_transformer_class,
    list_available_transformers,
    register_transformer,
)


class TestTransformerRegistry:
    """Test transformer registry functionality."""

    def test_transformers_auto_registered(self):
        """Test that transformers are automatically registered via decorators."""
        available_transformers = list_available_transformers()

        expected_transformers = {"standard_scale", "normalize", "log_scale"}
        assert expected_transformers.issubset(set(available_transformers))

    def test_get_transformer_class_valid(self):
        """Test getting transformer class by valid name."""
        cls = get_transformer_class("standard_scale")
        assert cls == StandardScaleTransformer

        cls = get_transformer_class("normalize")
        assert cls == NormalizeTransformer

        cls = get_transformer_class("log_scale")
        assert cls == LogTransformer

    def test_get_transformer_class_invalid(self):
        """Test getting transformer class by invalid name."""
        with pytest.raises(ValueError, match="Transformer 'invalid_name' not found"):
            get_transformer_class("invalid_name")

    def test_list_available_transformers_sorted(self):
        """Test that available transformers are returned sorted."""
        transformers = list_available_transformers()
        assert transformers == sorted(transformers)
        assert "standard_scale" in transformers
        assert "normalize" in transformers
        assert "log_scale" in transformers

    def test_create_sklearn_pipeline_single_transform(self):
        """Test creating pipeline with single transform."""
        pipeline = create_sklearn_pipeline(["standard_scale"], columns=["feature1"])

        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0][0] == "standard_scale"
        assert isinstance(pipeline.steps[0][1], StandardScaleTransformer)
        assert pipeline.steps[0][1].columns == ["feature1"]

    def test_create_sklearn_pipeline_multiple_transforms(self):
        """Test creating pipeline with multiple transforms."""
        pipeline = create_sklearn_pipeline(["standard_scale", "normalize"], columns=["feature1", "feature2"])

        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0][0] == "standard_scale"
        assert pipeline.steps[1][0] == "normalize"
        assert isinstance(pipeline.steps[0][1], StandardScaleTransformer)
        assert isinstance(pipeline.steps[1][1], NormalizeTransformer)

        # Check columns passed to both transformers
        for step_name, transformer in pipeline.steps:
            assert transformer.columns == ["feature1", "feature2"]

    def test_create_sklearn_pipeline_no_columns(self):
        """Test creating pipeline without specifying columns."""
        pipeline = create_sklearn_pipeline(["standard_scale"])

        assert isinstance(pipeline, Pipeline)
        assert pipeline.steps[0][1].columns is None

    def test_create_sklearn_pipeline_empty_transforms(self):
        """Test creating pipeline with empty transform list."""
        with pytest.raises(ValueError, match="At least one transform name must be provided"):
            create_sklearn_pipeline([])

    def test_create_sklearn_pipeline_invalid_transform(self):
        """Test creating pipeline with invalid transform name."""
        with pytest.raises(ValueError, match="Transformer 'invalid_transform' not found"):
            create_sklearn_pipeline(["standard_scale", "invalid_transform"])


class TestRegisterTransformerDecorator:
    """Test the register_transformer decorator."""

    def test_register_transformer_valid_name(self):
        """Test registering transformer with valid snake_case name."""

        @register_transformer("test_transformer")
        class TestTransformer(HydroTransformer):
            pass

        assert "test_transformer" in list_available_transformers()
        assert get_transformer_class("test_transformer") == TestTransformer

    def test_register_transformer_invalid_name_camelcase(self):
        """Test registering transformer with invalid camelCase name."""
        with pytest.raises(ValueError, match="must be snake_case"):

            @register_transformer("camelCaseName")
            class TestTransformer(HydroTransformer):
                pass

    def test_register_transformer_invalid_name_uppercase(self):
        """Test registering transformer with invalid uppercase name."""
        with pytest.raises(ValueError, match="must be snake_case"):

            @register_transformer("UPPERCASE_NAME")
            class TestTransformer(HydroTransformer):
                pass

    def test_register_transformer_invalid_name_dash(self):
        """Test registering transformer with invalid dash-separated name."""
        with pytest.raises(ValueError, match="must be snake_case"):

            @register_transformer("dash-separated-name")
            class TestTransformer(HydroTransformer):
                pass

    def test_register_transformer_invalid_name_spaces(self):
        """Test registering transformer with invalid name containing spaces."""
        with pytest.raises(ValueError, match="must be snake_case"):

            @register_transformer("name with spaces")
            class TestTransformer(HydroTransformer):
                pass

    def test_register_transformer_non_hydro_transformer(self):
        """Test registering non-HydroTransformer class."""
        with pytest.raises(ValueError, match="must inherit from HydroTransformer"):

            @register_transformer("invalid_class")
            class NotATransformer:
                pass
