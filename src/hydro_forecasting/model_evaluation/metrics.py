import numpy as np


def calculate_mse(pred: np.ndarray, obs: np.ndarray) -> float:
    """
    Calculate Mean Squared Error between predictions and observations.

    Args:
        pred: Predicted values array.
        obs: Observed values array.

    Returns:
        Mean squared error value, or NaN if input is empty.
    """
    if len(pred) == 0:
        return np.nan
    return np.mean((pred - obs) ** 2)


def calculate_mae(pred: np.ndarray, obs: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error between predictions and observations.

    Args:
        pred: Predicted values array.
        obs: Observed values array.

    Returns:
        Mean absolute error value, or NaN if input is empty.
    """
    if len(pred) == 0:
        return np.nan
    return np.mean(np.abs(pred - obs))


def calculate_rmse(pred: np.ndarray, obs: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error between predictions and observations.

    Args:
        pred: Predicted values array.
        obs: Observed values array.

    Returns:
        Root mean squared error value, or NaN if input is empty.
    """
    if len(pred) == 0:
        return np.nan
    return np.sqrt(np.mean((pred - obs) ** 2))


def calculate_nse(pred: np.ndarray, obs: np.ndarray) -> float:
    """
    Calculate Nash-Sutcliffe Efficiency coefficient.

    The NSE is a normalized statistic that determines the relative magnitude
    of the residual variance compared to the measured data variance.

    Args:
        pred: Predicted values array.
        obs: Observed values array.

    Returns:
        Nash-Sutcliffe efficiency value. Range: (-∞, 1], where 1 is perfect fit.
        Returns NaN if input is empty, -inf if denominator is zero with non-zero errors.
    """
    if len(obs) == 0 or len(pred) == 0:
        return np.nan
    mean_obs = np.mean(obs)
    if np.all(obs == mean_obs):
        return 1.0 if np.sum((pred - obs) ** 2) == 0 else -np.inf
    numerator = np.sum((pred - obs) ** 2)
    denominator = np.sum((obs - mean_obs) ** 2)
    if denominator == 0:
        return 1.0 if numerator == 0 else -np.inf
    return 1 - (numerator / denominator)


def calculate_pearson_r(pred: np.ndarray, obs: np.ndarray) -> float:
    """
    Calculate Pearson correlation coefficient between predictions and observations.

    Args:
        pred: Predicted values array.
        obs: Observed values array.

    Returns:
        Pearson correlation coefficient. Range: [-1, 1], or NaN if calculation fails.
    """
    if len(pred) < 2 or len(obs) < 2:
        return np.nan
    if np.std(obs) == 0 or np.std(pred) == 0:
        return np.nan
    return np.corrcoef(pred, obs)[0, 1]


def calculate_kge(pred: np.ndarray, obs: np.ndarray) -> float:
    """
    Calculate Kling-Gupta Efficiency.

    KGE is a goodness-of-fit measure that decomposes NSE into correlation,
    bias, and variability components.

    Args:
        pred: Predicted values array.
        obs: Observed values array.

    Returns:
        Kling-Gupta efficiency value. Range: (-∞, 1], where 1 is perfect fit.
        Returns NaN if input is empty or correlation cannot be calculated.
    """
    if len(obs) == 0 or len(pred) == 0:
        return np.nan
    mean_obs = np.mean(obs)
    if mean_obs == 0:
        return -np.inf
    r = calculate_pearson_r(pred, obs)
    if np.isnan(r):
        return np.nan
    beta = np.mean(pred) / mean_obs
    std_obs = np.std(obs)
    if std_obs == 0:
        return -np.inf
    gamma = np.std(pred) / std_obs
    return 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)


def calculate_pbias(pred: np.ndarray, obs: np.ndarray) -> float:
    """
    Calculate Percent Bias between predictions and observations.

    PBIAS measures the average tendency of predictions to be larger or smaller
    than observations. Positive values indicate model underestimation bias.

    Args:
        pred: Predicted values array.
        obs: Observed values array.

    Returns:
        Percent bias value. 0 indicates perfect model, positive values indicate
        underestimation, negative values indicate overestimation.
        Returns NaN if input is empty or sum of observations is zero.
    """
    if len(obs) == 0 or len(pred) == 0:
        return np.nan
    sum_obs = np.sum(obs)
    if sum_obs == 0:
        return np.nan
    return 100 * np.sum(pred - obs) / sum_obs


def calculate_atpe(pred: np.ndarray, obs: np.ndarray, percentile: float = 98.0) -> float:
    """
    Calculate Absolute Top Percentile Error.

    ATPE focuses on the model's ability to predict extreme values by calculating
    the mean absolute error for values above a specified percentile threshold.

    Args:
        pred: Predicted values array.
        obs: Observed values array.
        percentile: Percentile threshold for defining "top" values (default: 98.0).

    Returns:
        Mean absolute error for values above the percentile threshold.
        Returns NaN if input is empty or no values exceed the threshold.
    """
    if len(obs) == 0 or len(pred) == 0:
        return np.nan
    try:
        threshold = np.percentile(obs, percentile)
    except IndexError:
        return np.nan
    top_indices = obs >= threshold
    if not np.any(top_indices):
        return np.nan
    top_obs = obs[top_indices]
    top_pred = pred[top_indices]
    return np.mean(np.abs(top_pred - top_obs))
