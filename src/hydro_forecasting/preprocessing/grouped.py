from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import copy
import multiprocessing as mp
from functools import partial


def _process_groups_fit(
    pipeline_template: Pipeline,
    columns: List[str],
    df: pd.DataFrame,
    group_ids: List[str],
    group_identifier: str,
    y: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """Process a chunk of groups in fit mode.

    Args:
        pipeline_template: Template pipeline to clone for each group
        columns: Columns to transform
        df: DataFrame containing all data
        group_ids: List of group IDs to process in this chunk
        group_identifier: Column name identifying the grouping variable
        y: Target variable (optional, passed to pipeline fit method)

    Returns:
        Dictionary with fitted pipelines
    """
    from sklearn.base import clone

    result = {"fitted_pipelines": {}}

    for group_id in group_ids:
        # Filter data for this group
        group_mask = df[group_identifier] == group_id
        if not group_mask.any():
            continue

        group_data = df.loc[group_mask, columns].copy()

        # Handle target data if provided
        group_y = None
        if y is not None:
            group_y = y.loc[group_mask] if hasattr(y, "loc") else y[group_mask]

        # Clone the pipeline for this group
        group_pipeline = clone(pipeline_template)

        # Fit pipeline
        group_pipeline.fit(group_data, group_y)
        result["fitted_pipelines"][group_id] = group_pipeline

    return result


def _process_groups_transform(
    fitted_pipelines: Dict[str, Pipeline],
    columns: List[str],
    df: pd.DataFrame,
    group_ids: List[str],
    group_identifier: str,
) -> Dict[str, Any]:
    """Process a chunk of groups in transform mode.

    Args:
        fitted_pipelines: Dictionary of fitted pipelines for each group
        columns: Columns to transform
        df: DataFrame containing all data
        group_ids: List of group IDs to process in this chunk
        group_identifier: Column name identifying the grouping variable

    Returns:
        Dictionary with transformed data for each group
    """
    result = {"transformed_data": {}}

    for group_id in group_ids:
        if group_id not in fitted_pipelines:
            continue

        # Filter data for this group
        group_mask = df[group_identifier] == group_id
        if not group_mask.any():
            continue

        group_data = df.loc[group_mask, columns].copy()
        pipeline = fitted_pipelines[group_id]

        # Transform data
        transformed = pipeline.transform(group_data)
        result["transformed_data"][group_id] = transformed

    return result


def _process_groups_inverse_transform(
    fitted_pipelines: Dict[str, Pipeline],
    columns: List[str],
    df: pd.DataFrame,
    group_ids: List[str],
    group_identifier: str,
) -> Dict[str, Any]:
    """Process a chunk of groups in inverse_transform mode.

    Args:
        fitted_pipelines: Dictionary of fitted pipelines for each group
        columns: Columns to transform
        df: DataFrame containing all data
        group_ids: List of group IDs to process in this chunk
        group_identifier: Column name identifying the grouping variable

    Returns:
        Dictionary with inverse-transformed data for each group
    """
    result = {"transformed_data": {}}

    for group_id in group_ids:
        if group_id not in fitted_pipelines:
            continue

        # Filter data for this group
        group_mask = df[group_identifier] == group_id
        if not group_mask.any():
            continue

        group_data = df.loc[group_mask, columns].copy()
        pipeline = fitted_pipelines[group_id]

        # Check if pipeline has inverse_transform method
        if hasattr(pipeline, "inverse_transform"):
            inverse_data = pipeline.inverse_transform(group_data)
            result["transformed_data"][group_id] = inverse_data

    return result


class GroupedTransformer(BaseEstimator, TransformerMixin):
    """Applies transformations by group (e.g., by catchment).

    This transformer fits and applies a separate pipeline for each unique value in
    the group_identifier column. Useful for hydrological data where different
    catchments might require different preprocessing.

    Attributes:
        pipeline: sklearn Pipeline to apply to each group
        columns: Columns to transform
        group_identifier: Column name to group by
        fitted_pipelines: Dictionary mapping group values to fitted pipelines
        n_jobs: Number of parallel jobs to run (-1 for all cores)
        chunk_size: Number of groups to process in each chunk (None for auto)
    """

    def __init__(
        self,
        pipeline: Pipeline,
        columns: List[str],
        group_identifier: str,
        n_jobs: int = 1,
        chunk_size: Optional[int] = None,
    ):
        """Initialize GroupedTransformer.

        Args:
            pipeline: sklearn Pipeline to apply
            columns: Columns to transform
            group_identifier: Column name to group by
            n_jobs: Number of parallel jobs for multiprocessing (-1 for all cores)
            chunk_size: Number of groups to process in each chunk (None for auto)
        """
        self.pipeline = pipeline
        self.columns = columns
        self.group_identifier = group_identifier
        self.n_jobs = n_jobs
        self.chunk_size = chunk_size
        self.fitted_pipelines: Dict[Union[str, int], Pipeline] = {}
        self.all_groups = []  # Store all unique groups seen during fit

    def _split_groups_into_chunks(self, groups: List[str]) -> List[List[str]]:
        """Split groups into roughly equal-sized chunks for parallel processing."""
        # If not using multiprocessing, return all groups as a single chunk
        if self.n_jobs == 1:
            return [groups]

        # Determine number of jobs
        n_jobs = self.n_jobs
        if n_jobs < 0:
            n_jobs = mp.cpu_count()
        n_jobs = min(n_jobs, len(groups))

        if self.chunk_size:
            # Split based on specified chunk size
            return [
                groups[i : i + self.chunk_size]
                for i in range(0, len(groups), self.chunk_size)
            ]
        else:
            # Split based on number of jobs
            chunk_size = (len(groups) + n_jobs - 1) // n_jobs  # Ceiling division
            return [
                groups[i : i + chunk_size] for i in range(0, len(groups), chunk_size)
            ]

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "GroupedTransformer":
        """Fit a separate pipeline for each group.

        Args:
            X: Input data with group_identifier column
            y: Target variable (optional, passed to pipeline fit method)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If columns or group_identifier not found in data
        """
        # Validate columns
        missing_cols = [col for col in self.columns if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in data: {missing_cols}")
        if self.group_identifier not in X.columns:
            raise ValueError(
                f"Group identifier {self.group_identifier} not found in data"
            )

        # Store all unique groups to ensure we can handle new/unseen groups
        self.all_groups = X[self.group_identifier].unique().tolist()

        # If not using multiprocessing, fall back to original implementation
        if self.n_jobs == 1:
            for group in self.all_groups:
                group_mask = X[self.group_identifier] == group
                if not group_mask.any():
                    continue

                group_data = X.loc[group_mask, self.columns].copy()
                group_y = None
                if y is not None:
                    group_y = y.loc[group_mask] if hasattr(y, "loc") else y[group_mask]

                # Create a fresh copy of the pipeline for this group
                group_pipeline = copy.deepcopy(self.pipeline)
                group_pipeline.fit(group_data, group_y)
                self.fitted_pipelines[group] = group_pipeline
        else:
            # Use multiprocessing
            group_chunks = self._split_groups_into_chunks(self.all_groups)

            # Create process pool and process each chunk
            with mp.Pool(processes=self.n_jobs if self.n_jobs > 0 else None) as pool:
                process_func = partial(
                    _process_groups_fit,
                    self.pipeline,
                    self.columns,
                    X,
                    group_identifier=self.group_identifier,
                    y=y,
                )

                # Process all chunks in parallel
                results = pool.map(process_func, group_chunks)

            # Combine results from all processes
            for result in results:
                self.fitted_pipelines.update(result["fitted_pipelines"])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform each group with its fitted pipeline.

        Args:
            X: Input data with group_identifier column

        Returns:
            Transformed data

        Notes:
            Groups not seen during fit will be passed through unchanged
        """
        if self.group_identifier not in X.columns:
            raise ValueError(
                f"Group identifier {self.group_identifier} not found in data"
            )

        X_transformed = X.copy()

        # Get unique groups in the input data
        data_groups = X[self.group_identifier].unique().tolist()

        # Filter for groups we have fitted pipelines for
        groups_to_process = [g for g in data_groups if g in self.fitted_pipelines]

        # Warn about groups not seen during fit
        unseen_groups = [g for g in data_groups if g not in self.fitted_pipelines]
        if unseen_groups:
            print(
                f"Warning: Groups {unseen_groups} not seen during fit, passing through unchanged"
            )

        # If not using multiprocessing, fall back to original implementation
        if self.n_jobs == 1:
            # Process each group separately
            for group in groups_to_process:
                group_mask = X[self.group_identifier] == group
                if not group_mask.any():
                    continue

                group_data = X.loc[group_mask, self.columns].copy()

                # Transform this group's data
                transformed_data = self.fitted_pipelines[group].transform(group_data)

                # Handle the case where pipeline returns ndarray instead of DataFrame
                if isinstance(transformed_data, np.ndarray):
                    for i, col in enumerate(self.columns):
                        X_transformed.loc[group_mask, col] = transformed_data[:, i]
                else:
                    X_transformed.loc[group_mask, self.columns] = transformed_data
        else:
            # Use multiprocessing
            group_chunks = self._split_groups_into_chunks(groups_to_process)

            # Process chunks in parallel
            with mp.Pool(processes=self.n_jobs if self.n_jobs > 0 else None) as pool:
                process_func = partial(
                    _process_groups_transform,
                    self.fitted_pipelines,
                    self.columns,
                    X,
                    group_identifier=self.group_identifier,
                )
                results = pool.map(process_func, group_chunks)

            # Update the transformed DataFrame with results
            for result in results:
                for group_id, transformed_data in result["transformed_data"].items():
                    group_mask = X[self.group_identifier] == group_id
                    # Handle different return types from transformers
                    if isinstance(transformed_data, pd.DataFrame):
                        X_transformed.loc[group_mask, self.columns] = transformed_data
                    else:
                        # Assume numpy array
                        for i, col in enumerate(self.columns):
                            X_transformed.loc[group_mask, col] = transformed_data[:, i]

        return X_transformed

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform each group with its fitted pipeline.

        Args:
            X: Input data with group_identifier column

        Returns:
            Inverse transformed data

        Notes:
            Groups not seen during fit will be passed through unchanged
        """
        if self.group_identifier not in X.columns:
            raise ValueError(
                f"Group identifier {self.group_identifier} not found in data"
            )

        X_inverse = X.copy()

        # Get unique groups in the input data
        data_groups = X[self.group_identifier].unique().tolist()

        # Filter for groups we have fitted pipelines for
        groups_to_process = [g for g in data_groups if g in self.fitted_pipelines]

        # Warn about groups not seen during fit
        unseen_groups = [g for g in data_groups if g not in self.fitted_pipelines]
        if unseen_groups:
            print(
                f"Warning: Groups {unseen_groups} not seen during fit, passing through unchanged"
            )

        if self.n_jobs == 1:
            # Process each group separately
            for group in groups_to_process:
                group_mask = X[self.group_identifier] == group
                if not group_mask.any():
                    continue

                group_data = X.loc[group_mask, self.columns].copy()

                # Check if pipeline has inverse_transform method
                if not hasattr(self.fitted_pipelines[group], "inverse_transform"):
                    print(
                        f"Warning: Pipeline for group {group} does not support inverse_transform"
                    )
                    continue

                # Inverse transform this group's data
                inverse_data = self.fitted_pipelines[group].inverse_transform(
                    group_data
                )

                # Handle the case where pipeline returns ndarray instead of DataFrame
                if isinstance(inverse_data, np.ndarray):
                    for i, col in enumerate(self.columns):
                        X_inverse.loc[group_mask, col] = inverse_data[:, i]
                else:
                    X_inverse.loc[group_mask, self.columns] = inverse_data
        else:
            # Use multiprocessing
            group_chunks = self._split_groups_into_chunks(groups_to_process)

            # Process chunks in parallel
            with mp.Pool(processes=self.n_jobs if self.n_jobs > 0 else None) as pool:
                process_func = partial(
                    _process_groups_inverse_transform,
                    self.fitted_pipelines,
                    self.columns,
                    X,
                    group_identifier=self.group_identifier,
                )
                results = pool.map(process_func, group_chunks)

            # Update the inverse transformed DataFrame with results
            for result in results:
                for group_id, inverse_data in result["transformed_data"].items():
                    group_mask = X[self.group_identifier] == group_id
                    # Handle different return types from transformers
                    if isinstance(inverse_data, pd.DataFrame):
                        X_inverse.loc[group_mask, self.columns] = inverse_data
                    else:
                        # Assume numpy array
                        for i, col in enumerate(self.columns):
                            X_inverse.loc[group_mask, col] = inverse_data[:, i]

        return X_inverse

    def get_feature_names_out(self) -> List[str]:
        """Return feature names after transformation.

        Returns:
            List of output feature names
        """
        if not self.fitted_pipelines:
            raise ValueError("Transformer must be fitted before getting feature names")

        # Return the input columns as output feature names, assuming the pipeline
        # doesn't change the feature names
        return self.columns.copy()
