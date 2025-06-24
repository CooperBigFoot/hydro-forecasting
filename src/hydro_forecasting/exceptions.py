"""
Centralized exception hierarchy for the hydro-forecasting package.

This module defines all custom exceptions used throughout the application,
providing a consistent error handling interface and clear error categorization.
"""


class HydroForecastingError(Exception):
    """
    Base exception class for all hydro-forecasting related errors.

    This serves as the root of the exception hierarchy, allowing users
    to catch all package-specific errors with a single exception type.
    """

    pass


class ConfigurationError(HydroForecastingError):
    """
    Raised when there are configuration-related errors.

    This includes invalid configuration values, missing required settings,
    or configuration that fails validation checks.
    """

    pass


class FileOperationError(HydroForecastingError):
    """
    Raised when file operations fail.

    This includes file not found errors, permission issues, I/O errors,
    and other filesystem-related problems.
    """

    pass


class DataQualityError(HydroForecastingError):
    """
    Raised when data quality checks fail.

    This includes data validation failures, quality threshold violations,
    and other data integrity issues.
    """

    pass


class DataProcessingError(HydroForecastingError):
    """
    Raised when data processing operations fail.

    This includes data transformation errors, schema mismatches,
    and other data processing issues.
    """

    pass


class PipelineCompatibilityError(HydroForecastingError):
    """
    Raised when pipeline compatibility checks fail.

    This includes version mismatches, incompatible pipeline configurations,
    and other pipeline-related compatibility issues.
    """

    pass


class ModelTrainingError(HydroForecastingError):
    """
    Raised for errors during the model training, tuning, or fine-tuning lifecycle.

    This includes training failures, hyperparameter tuning errors, model
    convergence issues, and other problems that occur during the model
    development and optimization process.
    """

    pass
