"""Tests for pipeline builder functionality."""

import pytest

from hydro_forecasting.preprocessing import (
    GroupedPipeline,
    PipelineBuilder,
    PipelineSection,
    UnifiedPipeline,
)


class TestPipelineSection:
    """Test PipelineSection functionality."""

    def test_pipeline_section_initialization(self):
        """Test PipelineSection initialization."""
        builder = PipelineBuilder()
        section = PipelineSection(builder, "features")

        assert section.builder == builder
        assert section.section_name == "features"
        assert not section.is_complete()

    def test_transforms_valid(self):
        """Test setting valid transforms."""
        builder = PipelineBuilder()
        section = PipelineSection(builder, "features")

        result = section.transforms(["standard_scale", "normalize"])
        assert result == section  # Method chaining
        assert section._transforms == ["standard_scale", "normalize"]

    def test_transforms_empty_list(self):
        """Test setting empty transform list."""
        builder = PipelineBuilder()
        section = PipelineSection(builder, "features")

        with pytest.raises(ValueError, match="features: At least one transform must be specified"):
            section.transforms([])

    def test_transforms_invalid_name(self):
        """Test setting invalid transform name."""
        builder = PipelineBuilder()
        section = PipelineSection(builder, "features")

        with pytest.raises(ValueError, match="features: Invalid transform names: \\['invalid_transform'\\]"):
            section.transforms(["standard_scale", "invalid_transform"])

    def test_strategy_unified_valid(self):
        """Test setting valid unified strategy."""
        builder = PipelineBuilder()
        section = PipelineSection(builder, "features")

        result = section.strategy("unified", fit_on_n_basins=100)
        assert result == section  # Method chaining
        assert section._strategy == "unified"
        assert section._strategy_params == {"fit_on_n_basins": 100}

    def test_strategy_unified_no_params(self):
        """Test setting unified strategy without parameters."""
        builder = PipelineBuilder()
        section = PipelineSection(builder, "features")

        section.strategy("unified")
        assert section._strategy == "unified"
        assert section._strategy_params == {}

    def test_strategy_per_group_valid(self):
        """Test setting valid per_group strategy."""
        builder = PipelineBuilder()
        section = PipelineSection(builder, "target")

        result = section.strategy("per_group", group_by="gauge_id")
        assert result == section  # Method chaining
        assert section._strategy == "per_group"
        assert section._strategy_params == {"group_by": "gauge_id"}

    def test_strategy_invalid_name(self):
        """Test setting invalid strategy name."""
        builder = PipelineBuilder()
        section = PipelineSection(builder, "features")

        with pytest.raises(ValueError, match="features: Invalid strategy 'invalid_strategy'"):
            section.strategy("invalid_strategy")

    def test_strategy_unified_invalid_params(self):
        """Test setting unified strategy with invalid parameters."""
        builder = PipelineBuilder()
        section = PipelineSection(builder, "features")

        with pytest.raises(
            ValueError, match="features: Invalid parameters for unified strategy: \\{'invalid_param'\\}"
        ):
            section.strategy("unified", invalid_param="value")

    def test_strategy_per_group_invalid_params(self):
        """Test setting per_group strategy with invalid parameters."""
        builder = PipelineBuilder()
        section = PipelineSection(builder, "target")

        with pytest.raises(
            ValueError, match="target: Invalid parameters for per_group strategy: \\{'invalid_param'\\}"
        ):
            section.strategy("per_group", group_by="gauge_id", invalid_param="value")

    def test_columns_valid(self):
        """Test setting valid columns."""
        builder = PipelineBuilder()
        section = PipelineSection(builder, "features")

        result = section.columns(["feature1", "feature2"])
        assert result == builder  # Method chaining returns builder
        assert section._columns == ["feature1", "feature2"]

    def test_columns_empty_list(self):
        """Test setting empty column list."""
        builder = PipelineBuilder()
        section = PipelineSection(builder, "features")

        with pytest.raises(ValueError, match="features: Column list cannot be empty"):
            section.columns([])

    def test_is_complete_true(self):
        """Test is_complete when all required fields are set."""
        builder = PipelineBuilder()
        section = PipelineSection(builder, "features")

        section.transforms(["standard_scale"])
        section.strategy("unified")
        section.columns(["feature1"])

        assert section.is_complete()

    def test_is_complete_false(self):
        """Test is_complete when required fields are missing."""
        builder = PipelineBuilder()
        section = PipelineSection(builder, "features")

        # Missing all fields
        assert not section.is_complete()

        # Missing strategy and columns
        section.transforms(["standard_scale"])
        assert not section.is_complete()

        # Missing columns
        section.strategy("unified")
        assert not section.is_complete()

    def test_get_config_complete(self):
        """Test get_config when section is complete."""
        builder = PipelineBuilder()
        section = PipelineSection(builder, "features")

        section.transforms(["standard_scale", "normalize"])
        section.strategy("unified", fit_on_n_basins=50)
        section.columns(["feature1", "feature2"])

        config = section.get_config()
        expected = {
            "transforms": ["standard_scale", "normalize"],
            "strategy": "unified",
            "strategy_params": {"fit_on_n_basins": 50},
            "columns": ["feature1", "feature2"],
        }
        assert config == expected

    def test_get_config_incomplete(self):
        """Test get_config when section is incomplete."""
        builder = PipelineBuilder()
        section = PipelineSection(builder, "features")

        section.transforms(["standard_scale"])
        # Missing strategy and columns

        with pytest.raises(ValueError, match="features: Missing required configuration: \\['strategy', 'columns'\\]"):
            section.get_config()


class TestPipelineBuilder:
    """Test PipelineBuilder functionality."""

    def test_pipeline_builder_initialization(self):
        """Test PipelineBuilder initialization."""
        builder = PipelineBuilder()
        assert builder._sections == {}

    def test_features_section(self):
        """Test creating features section."""
        builder = PipelineBuilder()
        section = builder.features()

        assert isinstance(section, PipelineSection)
        assert section.section_name == "features"
        assert "features" in builder._sections
        assert builder._sections["features"] == section

    def test_target_section(self):
        """Test creating target section."""
        builder = PipelineBuilder()
        section = builder.target()

        assert isinstance(section, PipelineSection)
        assert section.section_name == "target"
        assert "target" in builder._sections
        assert builder._sections["target"] == section

    def test_static_features_section(self):
        """Test creating static_features section."""
        builder = PipelineBuilder()
        section = builder.static_features()

        assert isinstance(section, PipelineSection)
        assert section.section_name == "static_features"
        assert "static_features" in builder._sections
        assert builder._sections["static_features"] == section

    def test_build_no_sections(self):
        """Test building with no sections configured."""
        builder = PipelineBuilder()

        with pytest.raises(ValueError, match="No sections configured"):
            builder.build()

    def test_build_incomplete_sections(self):
        """Test building with incomplete sections."""
        builder = PipelineBuilder()
        builder.features().transforms(["standard_scale"])  # Missing strategy and columns

        with pytest.raises(ValueError, match="Incomplete sections: \\['features'\\]"):
            builder.build()

    def test_build_unified_pipeline_success(self):
        """Test successful build with unified pipeline."""
        builder = PipelineBuilder()

        builder.features().transforms(["standard_scale"]).strategy("unified", fit_on_n_basins=100).columns(["feature1"])

        config = builder.build()

        assert "features" in config
        features_config = config["features"]

        assert isinstance(features_config["pipeline"], UnifiedPipeline)
        assert features_config["strategy"] == "unified"
        assert features_config["columns"] == ["feature1"]
        assert features_config["fit_on_n_basins"] == 100

    def test_build_grouped_pipeline_success(self):
        """Test successful build with grouped pipeline."""
        builder = PipelineBuilder()

        builder.target().transforms(["standard_scale"]).strategy("per_group", group_by="gauge_id").columns(
            ["streamflow"]
        )

        config = builder.build()

        assert "target" in config
        target_config = config["target"]

        assert isinstance(target_config["pipeline"], GroupedPipeline)
        assert target_config["strategy"] == "per_group"
        assert target_config["columns"] == ["streamflow"]
        assert target_config["group_by"] == "gauge_id"

    def test_build_grouped_pipeline_missing_group_by(self):
        """Test build with grouped pipeline missing group_by parameter."""
        builder = PipelineBuilder()

        builder.target().transforms(["standard_scale"]).strategy("per_group").columns(["streamflow"])

        with pytest.raises(ValueError, match="target: 'group_by' parameter is required for per_group strategy"):
            builder.build()

    def test_build_multiple_sections(self):
        """Test building with multiple sections."""
        builder = PipelineBuilder()

        builder.features().transforms(["standard_scale"]).strategy("unified").columns(["feature1"])
        builder.target().transforms(["normalize"]).strategy("per_group", group_by="gauge_id").columns(["streamflow"])
        builder.static_features().transforms(["standard_scale"]).strategy("unified").columns(["static1"])

        config = builder.build()

        assert set(config.keys()) == {"features", "target", "static_features"}

        # Check features config
        assert isinstance(config["features"]["pipeline"], UnifiedPipeline)
        assert config["features"]["strategy"] == "unified"

        # Check target config
        assert isinstance(config["target"]["pipeline"], GroupedPipeline)
        assert config["target"]["strategy"] == "per_group"
        assert config["target"]["group_by"] == "gauge_id"

        # Check static_features config
        assert isinstance(config["static_features"]["pipeline"], UnifiedPipeline)
        assert config["static_features"]["strategy"] == "unified"

    def test_fluent_api_chaining(self):
        """Test fluent API method chaining works correctly."""
        config = (
            PipelineBuilder()
            .features()
            .transforms(["standard_scale", "normalize"])
            .strategy("unified", fit_on_n_basins=100)
            .columns(["feature1", "feature2"])
            .target()
            .transforms(["standard_scale"])
            .strategy("per_group", group_by="gauge_id")
            .columns(["streamflow"])
            .build()
        )

        assert set(config.keys()) == {"features", "target"}
        assert config["features"]["fit_on_n_basins"] == 100
        assert config["target"]["group_by"] == "gauge_id"
