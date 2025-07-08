"""Performance evaluation functions for model comparison and analysis."""

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .eval_utils import _parse_model_results, calculate_metric_statistics, extract_metric_values


def generate_performance_summary(
    results: dict[str, Any],
    metrics: list[str] | None = None,
    higher_is_better: dict[str, bool] | None = None,
    highlight_best: bool = True,
    decimal_places: int = 2,
) -> tuple[pd.DataFrame, str]:
    """
    Generate a performance summary table comparing models across architectures and variants.

    Creates a table where rows are grouped by architecture (e.g., LSTM, TSMixer) and variant
    (e.g., benchmark, regional), and columns show aggregated metrics (median ± std) computed
    across all basins and horizons.

    Args:
        results: Dictionary from TSForecastEvaluator with model results
        metrics: List of metrics to include (e.g., ["nse", "kge", "rmse", "mae"]).
                If None, defaults to ["nse", "kge", "rmse", "mae", "mse", "pearson_r", "pbias"]
        higher_is_better: Dictionary mapping metric names to whether higher values are better.
                         If None, uses sensible defaults.
        highlight_best: Whether to highlight the best variant for each architecture/metric
        decimal_places: Number of decimal places for formatting (default: 2)

    Returns:
        Tuple of (DataFrame, LaTeX string):
        - DataFrame with multi-index (Architecture, Variant) and metrics as columns
        - LaTeX table code ready for inclusion in reports

    Example:
        >>> summary_df, latex_code = generate_performance_summary(
        ...     results,
        ...     metrics=["nse", "kge", "rmse"],
        ...     higher_is_better={"nse": True, "kge": True, "rmse": False}
        ... )
    """
    # Default metrics if not specified
    if metrics is None:
        metrics = ["nse", "kge", "rmse", "mae", "mse", "pearson_r", "pbias"]

    # Default higher_is_better mapping
    if higher_is_better is None:
        higher_is_better = {
            "nse": True,
            "kge": True,
            "pearson_r": True,
            "rmse": False,
            "mae": False,
            "mse": False,
            "pbias": False,  # Closer to 0 is better, but we'll use absolute value
            "atpe": False,
        }

    # Parse model names and extract data
    model_data = {}
    for model_name, model_result in results.items():
        if "_" not in model_name or "metrics_by_gauge" not in model_result:
            continue

        # Extract architecture and variant
        parts = model_name.split("_", 1)
        arch = parts[0].upper()  # Convert to uppercase for consistency
        variant = parts[1]

        if arch not in model_data:
            model_data[arch] = {}

        # Collect all metric values across all basins and horizons
        metric_values = {metric: [] for metric in metrics}
        metrics_by_gauge = model_result["metrics_by_gauge"]

        for gauge_data in metrics_by_gauge.values():
            for horizon_data in gauge_data.values():
                for metric in metrics:
                    if metric in horizon_data:
                        value = horizon_data[metric]
                        if not np.isnan(value):
                            metric_values[metric].append(value)

        # Calculate statistics
        model_data[arch][variant] = {}
        for metric in metrics:
            if metric_values[metric]:
                values = np.array(metric_values[metric])
                median = np.median(values)
                std = np.std(values)
                model_data[arch][variant][metric] = {"median": median, "std": std, "count": len(values)}

    # First, identify best variants for highlighting and statistical testing

    # Identify best variants for highlighting
    best_variants = {}
    if highlight_best:
        for arch in model_data:
            best_variants[arch] = {}
            for metric in metrics:
                best_value = None
                best_variant = None

                for variant in model_data[arch]:
                    if metric in model_data[arch][variant]:
                        if metric == "pbias":
                            # For PBIAS, best is closest to 0 (minimum absolute value)
                            value = abs(model_data[arch][variant][metric]["median"])
                        else:
                            value = model_data[arch][variant][metric]["median"]

                        is_better = higher_is_better.get(metric, True)

                        if (
                            best_value is None
                            or (is_better and value > best_value)
                            or (not is_better and value < best_value)
                        ):
                            best_value = value
                            best_variant = variant

                if best_variant is not None:
                    best_variants[arch][metric] = best_variant

    # Collect paired performance data by basin for statistical testing
    # For each basin, we calculate the median performance metric across all forecast horizons
    # to get a single representative value per basin for the statistical test
    paired_data = {}
    for arch in model_data:
        paired_data[arch] = {}
        for metric in metrics:
            paired_data[arch][metric] = {}
            
            # Get list of all basins that have data for all variants
            all_basins = set()
            variant_basins = {}
            
            for variant in model_data[arch]:
                variant_basins[variant] = set()
                if "metrics_by_gauge" in results.get(f"{arch.lower()}_{variant}", {}):
                    metrics_by_gauge = results[f"{arch.lower()}_{variant}"]["metrics_by_gauge"]
                    for gauge_id, gauge_data in metrics_by_gauge.items():
                        # Check if this gauge has the metric for any horizon
                        has_metric = False
                        for horizon_data in gauge_data.values():
                            if metric in horizon_data and not np.isnan(horizon_data[metric]):
                                has_metric = True
                                break
                        if has_metric:
                            variant_basins[variant].add(gauge_id)
                            all_basins.add(gauge_id)
            
            # Find basins common to all variants
            common_basins = None
            for variant, basins in variant_basins.items():
                if common_basins is None:
                    common_basins = basins.copy()
                else:
                    common_basins = common_basins.intersection(basins)
            
            if not common_basins:
                continue
            
            # Collect paired values for each variant
            for variant in model_data[arch]:
                values = []
                if f"{arch.lower()}_{variant}" in results:
                    metrics_by_gauge = results[f"{arch.lower()}_{variant}"].get("metrics_by_gauge", {})
                    
                    for gauge_id in sorted(common_basins):  # Sort for consistent ordering
                        gauge_values = []
                        if gauge_id in metrics_by_gauge:
                            for horizon_data in metrics_by_gauge[gauge_id].values():
                                if metric in horizon_data and not np.isnan(horizon_data[metric]):
                                    gauge_values.append(horizon_data[metric])
                        
                        # Use median across horizons for this gauge
                        if gauge_values:
                            values.append(np.median(gauge_values))
                    
                paired_data[arch][metric][variant] = np.array(values)

    # Perform statistical significance testing
    significance_markers = {}
    for arch in model_data:
        significance_markers[arch] = {}
        for metric in metrics:
            significance_markers[arch][metric] = {}
            
            # Skip if no best variant identified
            if arch not in best_variants or metric not in best_variants[arch]:
                continue
                
            best_variant = best_variants[arch][metric]
            
            # Skip if no paired data available
            if (arch not in paired_data or metric not in paired_data[arch] or 
                best_variant not in paired_data[arch][metric]):
                continue
            
            best_values = paired_data[arch][metric][best_variant]
            
            # Compare each variant to the best
            for variant in model_data[arch]:
                if variant == best_variant:
                    significance_markers[arch][metric][variant] = False  # Best variant doesn't get asterisk
                    continue
                
                if variant not in paired_data[arch][metric]:
                    significance_markers[arch][metric][variant] = False
                    continue
                
                variant_values = paired_data[arch][metric][variant]
                
                # Ensure same number of paired observations
                if len(best_values) != len(variant_values) or len(best_values) < 5:
                    # Need at least 5 pairs for Wilcoxon test
                    significance_markers[arch][metric][variant] = False
                    continue
                
                # Check if samples are identical to avoid Wilcoxon test errors
                if np.array_equal(best_values, variant_values):
                    # Identical samples mean p-value = 1.0 (definitely not different)
                    significance_markers[arch][metric][variant] = True
                else:
                    try:
                        # Perform Wilcoxon signed-rank test
                        _, p_value = wilcoxon(best_values, variant_values)
                        
                        # Mark with asterisk if NOT significantly different (p >= 0.05)
                        significance_markers[arch][metric][variant] = p_value >= 0.05
                    except Exception:
                        # If test fails for any reason, don't mark with asterisk
                        significance_markers[arch][metric][variant] = False

    # Now create DataFrame with all the calculated information
    rows = []
    all_medians_by_metric = {metric: [] for metric in metrics}  # Track all medians for overall calculation

    for arch in sorted(model_data.keys()):
        for variant in sorted(model_data[arch].keys()):
            row_data = {"Architecture": arch, "Variant": variant}

            for metric in metrics:
                if metric in model_data[arch][variant]:
                    stats = model_data[arch][variant][metric]
                    median = stats["median"]
                    std = stats["std"]

                    # Track median for overall calculation
                    all_medians_by_metric[metric].append(median)

                    # Format the value
                    if metric == "pbias":
                        # For PBIAS, use absolute value for comparison but show actual value
                        row_data[f"{metric}_raw"] = median
                        row_data[f"{metric}_abs"] = abs(median)
                    else:
                        row_data[f"{metric}_raw"] = median

                    # Add asterisk if not significantly different from best
                    value_str = f"{median:.{decimal_places}f} ± {std:.{decimal_places}f}"
                    if (arch in significance_markers and
                        metric in significance_markers[arch] and
                        variant in significance_markers[arch][metric] and
                        significance_markers[arch][metric][variant]):
                        value_str += "*"

                    row_data[metric] = value_str
                else:
                    row_data[metric] = "N/A"
                    row_data[f"{metric}_raw"] = np.nan

            rows.append(row_data)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Calculate overall median per variant
    variant_medians = {}
    all_variants = sorted({variant for arch_data in model_data.values() for variant in arch_data})

    for variant in all_variants:
        variant_medians[variant] = {metric: [] for metric in metrics}

        # Collect medians for this variant across all architectures
        for arch in model_data:
            if variant in model_data[arch]:
                for metric in metrics:
                    if metric in model_data[arch][variant]:
                        variant_medians[variant][metric].append(model_data[arch][variant][metric]["median"])

    # Add overall rows for each variant (without statistical testing)
    overall_rows = []
    for variant in all_variants:
        overall_row = {"Architecture": "OVERALL", "Variant": variant}

        for metric in metrics:
            if variant_medians[variant][metric]:
                median_of_medians = np.median(variant_medians[variant][metric])
                std_of_medians = np.std(variant_medians[variant][metric])
                
                # Format without asterisk (no statistical testing for OVERALL)
                value_str = f"{median_of_medians:.{decimal_places}f} ± {std_of_medians:.{decimal_places}f}"
                
                overall_row[metric] = value_str
                overall_row[f"{metric}_raw"] = median_of_medians
            else:
                overall_row[metric] = "N/A"
                overall_row[f"{metric}_raw"] = np.nan

        overall_rows.append(overall_row)

    # Append overall rows to DataFrame
    df = pd.concat([df, pd.DataFrame(overall_rows)], ignore_index=True)

    # Apply highlighting to DataFrame (including OVERALL rows)
    if highlight_best:
        # First, identify best OVERALL variant for each metric
        best_overall_variants = {}
        for metric in metrics:
            best_value = None
            best_variant = None

            # Check each OVERALL row
            for _, row in df[df["Architecture"] == "OVERALL"].iterrows():
                variant = row["Variant"]
                if f"{metric}_raw" in row and not pd.isna(row[f"{metric}_raw"]):
                    value = row[f"{metric}_raw"]

                    if metric == "pbias":
                        value = abs(value)

                    is_better = higher_is_better.get(metric, True)

                    if (
                        best_value is None
                        or (is_better and value > best_value)
                        or (not is_better and value < best_value)
                    ):
                        best_value = value
                        best_variant = variant

            if best_variant is not None:
                best_overall_variants[metric] = best_variant

        # Apply highlighting to all rows
        for idx, row in df.iterrows():
            arch = row["Architecture"]
            variant = row["Variant"]

            if arch == "OVERALL":
                # Highlight best OVERALL variants
                for metric in metrics:
                    if metric in best_overall_variants and best_overall_variants[metric] == variant:
                        # Add bold markers for best values
                        df.at[idx, metric] = "**" + df.at[idx, metric] + "**"
            else:
                # Highlight best within each architecture
                for metric in metrics:
                    if (
                        arch in best_variants
                        and metric in best_variants[arch]
                        and best_variants[arch][metric] == variant
                    ):
                        # Add bold markers for best values
                        df.at[idx, metric] = "**" + df.at[idx, metric] + "**"

    # Create LaTeX table
    latex_rows = []

    # Header
    latex_rows.append("\\begin{table}[htbp]")
    latex_rows.append("\\centering")
    latex_rows.append("\\caption{Model Performance Summary}")

    # Calculate number of columns
    num_metrics = len(metrics)
    col_spec = "ll" + "r" * num_metrics  # Architecture, Variant, then metrics

    latex_rows.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_rows.append("\\toprule")

    # Header row
    metric_headers = " & ".join([f"\\textbf{{{m.upper()}}}" for m in metrics])
    latex_rows.append(f"\\textbf{{Architecture}} & \\textbf{{Variant}} & {metric_headers} \\\\")
    latex_rows.append("\\midrule")

    # Data rows
    current_arch = None
    for _, row in df.iterrows():
        arch = row["Architecture"]
        variant = row["Variant"]

        # Add separator before OVERALL section
        if arch == "OVERALL" and current_arch != "OVERALL":
            latex_rows.append("\\midrule")

        # Format architecture (only show once per group)
        if arch != current_arch:
            arch_str = f"\\textbf{{{arch}}}"
            current_arch = arch
        else:
            arch_str = ""

        # Format metrics
        metric_values = []
        for metric in metrics:
            value = row[metric]

            # Remove markdown bold indicators for LaTeX processing
            value_clean = value.replace("**", "")

            # Add bold for best values in LaTeX
            if arch == "OVERALL":
                # Check if this is the best OVERALL variant
                if metric in best_overall_variants and best_overall_variants[metric] == variant:
                    value_formatted = f"\\textbf{{{value_clean}}}"
                else:
                    value_formatted = value_clean
            elif (
                arch in best_variants
                and metric in best_variants[arch]
                and best_variants[arch][metric] == variant
            ):
                value_formatted = f"\\textbf{{{value_clean}}}"
            else:
                value_formatted = value_clean

            metric_values.append(value_formatted)

        metric_str = " & ".join(metric_values)
        latex_rows.append(f"{arch_str} & {variant} & {metric_str} \\\\")

    # Footer
    latex_rows.append("\\bottomrule")
    latex_rows.append("\\end{tabular}")
    latex_rows.append("\\vspace{0.1cm}")
    latex_rows.append(
        "\\footnotesize{\\textbf{Note:} Bold values indicate the best performance for each architecture and metric. "
        "An asterisk (*) indicates that a model's performance is not statistically significantly different "
        "from the best-performing model (p ≥ 0.05, Wilcoxon signed-rank test).}"
    )
    latex_rows.append("\\label{tab:performance_summary}")
    latex_rows.append("\\end{table}")

    latex_code = "\n".join(latex_rows)

    # Clean up DataFrame (remove raw columns)
    display_columns = ["Architecture", "Variant"] + metrics
    df_display = df[display_columns].copy()

    return df_display, latex_code


def print_performance_summary(
    results: dict[str, Any],
    architectures: list[str] | None = None,
    variants: list[str] | None = None,
    metric: str = "nse",
    horizons: list[int] | None = None,
) -> None:
    """
    Print a formatted performance summary for specified architectures, variants, and metric.

    Shows median ± 1 standard deviation across all basins for each horizon.

    Args:
        results: Dictionary from TSForecastEvaluator with model results
        architectures: List of architectures to include (e.g., ["ealstm", "tft"]).
                      If None, includes all available architectures.
        variants: List of variants to include (e.g., ["benchmark", "finetuned"]).
                 If None, includes all available variants.
        metric: Performance metric to display (default: "nse")
        horizons: List of horizons to include. If None, includes all available horizons.

    Example:
        >>> print_performance_summary(
        ...     results,
        ...     architectures=["ealstm", "tft"],
        ...     variants=["benchmark", "finetuned"],
        ...     metric="nse"
        ... )
        Performance Summary for NSE:
        ==========================================
        EALSTM - benchmark:
          Horizon 1: 0.75 ± 0.12 (n=100 basins)
          Horizon 5: 0.68 ± 0.15 (n=100 basins)
          ...
    """
    # Parse model results
    model_data = _parse_model_results(results)

    # Auto-detect architectures and variants if not provided
    architectures = sorted(model_data.keys()) if architectures is None else [arch.lower() for arch in architectures]

    if variants is None:
        all_variants = set()
        for arch_data in model_data.values():
            all_variants.update(arch_data.keys())
        variants = sorted(all_variants)

    # Print header
    print(f"\nPerformance Summary for {metric.upper()}:")
    print("=" * 50)

    # Process each architecture and variant combination
    for arch in architectures:
        if arch not in model_data:
            print(f"\nWarning: Architecture '{arch}' not found in results")
            continue

        for variant in variants:
            if variant not in model_data[arch]:
                continue

            print(f"\n{arch.upper()} - {variant}:")

            # Get model result
            model_result = model_data[arch][variant]
            if "metrics_by_gauge" not in model_result:
                print("  No metrics data available")
                continue

            # Extract metric values for all horizons
            metrics_by_gauge = model_result["metrics_by_gauge"]
            horizon_values = extract_metric_values(metrics_by_gauge, metric, horizons)

            if not horizon_values:
                print(f"  No data available for metric '{metric}'")
                continue

            # Sort horizons for display
            sorted_horizons = sorted(horizon_values.keys())

            # Calculate and display statistics for each horizon
            for horizon in sorted_horizons:
                values = horizon_values[horizon]
                stats = calculate_metric_statistics(values)

                if stats["count"] > 0:
                    print(
                        f"  Horizon {horizon:2d}: {stats['median']:.3f} ± {stats['std']:.3f} "
                        f"(n={stats['count']} basins)"
                    )
                else:
                    print(f"  Horizon {horizon:2d}: No data")

    print("=" * 50)
