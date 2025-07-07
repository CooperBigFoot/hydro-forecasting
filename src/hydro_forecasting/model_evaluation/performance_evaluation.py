"""Performance evaluation functions for model comparison and analysis."""

from typing import Any

import numpy as np
import pandas as pd

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

    # Create DataFrame
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

                    row_data[metric] = f"{median:.{decimal_places}f} ± {std:.{decimal_places}f}"
                else:
                    row_data[metric] = "N/A"
                    row_data[f"{metric}_raw"] = np.nan

            rows.append(row_data)

    # Create DataFrame with multi-index
    df = pd.DataFrame(rows)

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

    # Create DataFrame first (without overall row)
    df = pd.DataFrame(rows)

    # Apply highlighting to DataFrame
    if highlight_best:
        for idx, row in df.iterrows():
            arch = row["Architecture"]
            variant = row["Variant"]

            if arch != "OVERALL":
                # Highlight best within each architecture
                for metric in metrics:
                    if arch in best_variants and metric in best_variants[arch]:
                        if best_variants[arch][metric] == variant:
                            # Add bold markers for best values
                            df.at[idx, metric] = "**" + df.at[idx, metric] + "**"

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

    # Add overall rows for each variant
    overall_rows = []
    for variant in all_variants:
        overall_row = {"Architecture": "OVERALL", "Variant": variant}

        for metric in metrics:
            if variant_medians[variant][metric]:
                median_of_medians = np.median(variant_medians[variant][metric])
                std_of_medians = np.std(variant_medians[variant][metric])
                overall_row[metric] = f"{median_of_medians:.{decimal_places}f} ± {std_of_medians:.{decimal_places}f}"
                overall_row[f"{metric}_raw"] = median_of_medians
            else:
                overall_row[metric] = "N/A"
                overall_row[f"{metric}_raw"] = np.nan

        overall_rows.append(overall_row)

    # Append overall rows to DataFrame
    df = pd.concat([df, pd.DataFrame(overall_rows)], ignore_index=True)

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
            if (
                arch != "OVERALL"
                and arch in best_variants
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
        "\\footnotesize{\\textbf{Note:} Bold values indicate the best performance for each architecture and metric.}"
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
