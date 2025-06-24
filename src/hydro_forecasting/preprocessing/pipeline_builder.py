"""Fluent API builder for hydro forecasting preprocessing pipelines."""

from __future__ import annotations
from typing import Any

from .grouped import GroupedPipeline
from .transformer_registry import create_sklearn_pipeline, list_available_transformers
from .unified import UnifiedPipeline


class PipelineSection:
    """Represents a section of the preprocessing pipeline configuration."""
    
    def __init__(self, builder: PipelineBuilder, section_name: str):
        """Initialize pipeline section.
        
        Args:
            builder: Parent builder instance
            section_name: Name of this section (features, target, static_features)
        """
        self.builder = builder
        self.section_name = section_name
        self._transforms: list[str] | None = None
        self._strategy: str | None = None
        self._strategy_params: dict[str, Any] = {}
        self._columns: list[str] | None = None
    
    def transforms(self, transform_list: list[str]) -> PipelineSection:
        """Set the list of transforms for this section.
        
        Args:
            transform_list: List of transform names to apply
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If any transform name is not available
        """
        if not transform_list:
            raise ValueError(f"{self.section_name}: At least one transform must be specified")
        
        # Validate all transform names exist
        available_transforms = list_available_transformers()
        invalid_transforms = [t for t in transform_list if t not in available_transforms]
        if invalid_transforms:
            raise ValueError(
                f"{self.section_name}: Invalid transform names: {invalid_transforms}. "
                f"Available transforms: {available_transforms}"
            )
        
        self._transforms = transform_list
        return self
    
    def strategy(self, strategy_name: str, **kwargs) -> PipelineSection:
        """Set the strategy for this section.
        
        Args:
            strategy_name: Either "per_group" or "unified"
            **kwargs: Strategy-specific parameters
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If strategy name is invalid or parameters are incorrect
        """
        valid_strategies = ["per_group", "unified"]
        if strategy_name not in valid_strategies:
            raise ValueError(
                f"{self.section_name}: Invalid strategy '{strategy_name}'. "
                f"Valid strategies: {valid_strategies}"
            )
        
        # Validate strategy-specific parameters
        if strategy_name == "unified":
            # fit_on_n_basins is allowed for unified strategy
            valid_params = {"fit_on_n_basins"}
            invalid_params = set(kwargs.keys()) - valid_params
            if invalid_params:
                raise ValueError(
                    f"{self.section_name}: Invalid parameters for unified strategy: {invalid_params}. "
                    f"Valid parameters: {valid_params}"
                )
        elif strategy_name == "per_group":
            # group_by is allowed for per_group strategy
            valid_params = {"group_by"}
            invalid_params = set(kwargs.keys()) - valid_params
            if invalid_params:
                raise ValueError(
                    f"{self.section_name}: Invalid parameters for per_group strategy: {invalid_params}. "
                    f"Valid parameters: {valid_params}"
                )
        
        self._strategy = strategy_name
        self._strategy_params = kwargs
        return self
    
    def columns(self, column_list: list[str]) -> PipelineBuilder:
        """Set the columns for this section.
        
        Args:
            column_list: List of column names to transform
            
        Returns:
            Builder for method chaining
            
        Raises:
            ValueError: If column list is empty
        """
        if not column_list:
            raise ValueError(f"{self.section_name}: Column list cannot be empty")
        
        self._columns = column_list
        return self.builder
    
    def is_complete(self) -> bool:
        """Check if this section has all required configuration."""
        return (
            self._transforms is not None and
            self._strategy is not None and
            self._columns is not None
        )
    
    def get_config(self) -> dict[str, Any]:
        """Get the configuration for this section."""
        if not self.is_complete():
            missing = []
            if self._transforms is None:
                missing.append("transforms")
            if self._strategy is None:
                missing.append("strategy")
            if self._columns is None:
                missing.append("columns")
            raise ValueError(f"{self.section_name}: Missing required configuration: {missing}")
        
        return {
            "transforms": self._transforms,
            "strategy": self._strategy,
            "strategy_params": self._strategy_params,
            "columns": self._columns
        }


class PipelineBuilder:
    """Fluent API builder for preprocessing pipeline configurations."""
    
    def __init__(self):
        """Initialize the pipeline builder."""
        self._sections: dict[str, PipelineSection] = {}
    
    def features(self) -> PipelineSection:
        """Configure the features section.
        
        Returns:
            PipelineSection for features configuration
        """
        section = PipelineSection(self, "features")
        self._sections["features"] = section
        return section
    
    def target(self) -> PipelineSection:
        """Configure the target section.
        
        Returns:
            PipelineSection for target configuration
        """
        section = PipelineSection(self, "target")
        self._sections["target"] = section
        return section
    
    def static_features(self) -> PipelineSection:
        """Configure the static_features section.
        
        Returns:
            PipelineSection for static_features configuration
        """
        section = PipelineSection(self, "static_features")
        self._sections["static_features"] = section
        return section
    
    def build(self) -> dict[str, dict[str, Any]]:
        """Build the final preprocessing configuration.
        
        Returns:
            Dictionary compatible with existing DataModule preprocessing_configs parameter
            
        Raises:
            ValueError: If any configured section is incomplete
        """
        if not self._sections:
            raise ValueError("No sections configured. Use features(), target(), or static_features() to configure sections.")
        
        # Validate all sections are complete
        incomplete_sections = [name for name, section in self._sections.items() if not section.is_complete()]
        if incomplete_sections:
            raise ValueError(f"Incomplete sections: {incomplete_sections}")
        
        # Build configuration for each section
        config = {}
        for section_name, section in self._sections.items():
            config[section_name] = self._build_section_config(section)
        
        return config
    
    def _build_section_config(self, section: PipelineSection) -> dict[str, Any]:
        """Build configuration for a specific section.
        
        Args:
            section: PipelineSection to build config for
            
        Returns:
            Section-specific configuration dictionary
        """
        section_config = section.get_config()
        
        # Create sklearn pipeline from transforms
        sklearn_pipeline = create_sklearn_pipeline(
            section_config["transforms"],
            columns=section_config["columns"]
        )
        
        # Wrap in appropriate pipeline wrapper
        if section_config["strategy"] == "unified":
            wrapped_pipeline = UnifiedPipeline(
                pipeline=sklearn_pipeline,
                columns=section_config["columns"]
            )
        else:  # per_group
            # group_by is required for per_group strategy
            if "group_by" not in section_config["strategy_params"]:
                raise ValueError(f"{section.section_name}: 'group_by' parameter is required for per_group strategy")
            
            wrapped_pipeline = GroupedPipeline(
                pipeline=sklearn_pipeline,
                columns=section_config["columns"],
                group_identifier=section_config["strategy_params"]["group_by"]
            )
        
        # Build final configuration
        result = {
            "pipeline": wrapped_pipeline,
            "strategy": section_config["strategy"],
            "columns": section_config["columns"]
        }
        
        # Add strategy-specific parameters
        result.update(section_config["strategy_params"])
        
        return result