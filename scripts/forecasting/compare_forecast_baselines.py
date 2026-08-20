"""
EirGrid wind-forecast model comparison.

Compares four forecasting approaches:

1. Persistence
2. Linear Regression
3. Feature-Engineered Linear Regression
4. Direct Multi-Horizon Linear Regression

Primary metric:
    NMAE (%)

Lower NMAE is better.

The script reads benchmark CSV files from:

    data/processed/

and produces:

    data/processed/forecast_model_comparison.csv

The comparison is explicitly renamed before merging so that
model-specific metric columns are deterministic and cannot
depend on pandas suffix behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]

PROCESSED = ROOT / "data" / "processed"

PERSISTENCE_FILE = (
    PROCESSED / "persistence_forecast_benchmark.csv"
)

LINEAR_REGRESSION_FILE = (
    PROCESSED / "linear_regression_forecast_benchmark.csv"
)

FEATURE_ENGINEERED_FILE = (
    PROCESSED
    / "feature_engineered_linear_regression_forecast_benchmark.csv"
)

DIRECT_MULTI_HORIZON_FILE = (
    PROCESSED
    / "direct_multi_horizon_linear_regression_forecast_benchmark.csv"
)

OUTPUT_FILE = (
    PROCESSED / "forecast_model_comparison.csv"
)


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_CONFIG = {
    "persistence": {
        "label": "Persistence",
        "file": PERSISTENCE_FILE,
    },
    "linear_regression": {
        "label": "Linear Regression",
        "file": LINEAR_REGRESSION_FILE,
    },
    "feature_engineered_linear_regression": {
        "label": "Feature-Engineered Linear Regression",
        "file": FEATURE_ENGINEERED_FILE,
    },
    "direct_multi_horizon_linear_regression": {
        "label": "Direct Multi-Horizon Linear Regression",
        "file": DIRECT_MULTI_HORIZON_FILE,
    },
}


REQUIRED_COLUMNS = {
    "horizon",
    "horizon_steps",
    "samples",
    "mae_mw",
    "rmse_mw",
    "nmae",
    "nmae_percent",
}


# ============================================================================
# LOAD ONE BENCHMARK
# ============================================================================


def load_benchmark(
    path: Path,
    model_key: str,
    model_label: str,
) -> pd.DataFrame:
    """
    Load and validate one benchmark CSV.

    Model-specific metric columns are renamed immediately.

    Example:

        nmae_percent

    becomes:

        nmae_percent_persistence

    or:

        nmae_percent_direct_multi_horizon_linear_regression

    This prevents merge/suffix naming problems.
    """

    print(
        f"Loading {model_label} benchmark..."
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark file not found:\n{path}"
        )

    dataframe = pd.read_csv(path)

    missing = (
        REQUIRED_COLUMNS
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: "
            f"{sorted(missing)}"
        )

    dataframe = dataframe.copy()

    # ------------------------------------------------------------
    # Validate horizon uniqueness.
    # ------------------------------------------------------------

    if dataframe[
        ["horizon", "horizon_steps"]
    ].duplicated().any():

        raise ValueError(
            f"{path.name} contains duplicate forecast horizons."
        )

    # ------------------------------------------------------------
    # Validate numeric metric columns.
    # ------------------------------------------------------------

    numeric_columns = [
        "horizon_steps",
        "samples",
        "mae_mw",
        "rmse_mw",
        "nmae",
        "nmae_percent",
    ]

    for column in numeric_columns:

        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="raise",
        )

    # ------------------------------------------------------------
    # Explicitly rename model-specific columns.
    # ------------------------------------------------------------

    rename_map = {
        "samples": f"samples_{model_key}",
        "mae_mw": f"mae_mw_{model_key}",
        "rmse_mw": f"rmse_mw_{model_key}",
        "nmae": f"nmae_{model_key}",
        "nmae_percent": f"nmae_percent_{model_key}",
    }

    dataframe = dataframe.rename(
        columns=rename_map
    )

    # ------------------------------------------------------------
    # Keep only the columns needed by the comparison.
    # ------------------------------------------------------------

    dataframe = dataframe[
        [
            "horizon",
            "horizon_steps",
            f"samples_{model_key}",
            f"mae_mw_{model_key}",
            f"rmse_mw_{model_key}",
            f"nmae_{model_key}",
            f"nmae_percent_{model_key}",
        ]
    ].copy()

    print(
        f"  Horizons: {len(dataframe)}"
    )

    return dataframe


# ============================================================================
# MERGE ALL BENCHMARKS
# ============================================================================


def build_comparison_table(
    benchmarks: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Merge all benchmark results by forecast horizon.
    """

    comparison: pd.DataFrame | None = None

    for dataframe in benchmarks.values():

        if comparison is None:

            comparison = dataframe.copy()

        else:

            comparison = comparison.merge(
                dataframe,
                on=[
                    "horizon",
                    "horizon_steps",
                ],
                how="inner",
                validate="one_to_one",
            )

    if comparison is None or comparison.empty:

        raise ValueError(
            "No common forecast horizons were found "
            "across the benchmark files."
        )

    # ------------------------------------------------------------
    # Expected horizons.
    # ------------------------------------------------------------

    expected_horizons = {
        "15min",
        "30min",
        "1hour",
        "2hour",
        "4hour",
    }

    actual_horizons = set(
        comparison["horizon"]
    )

    missing_horizons = (
        expected_horizons
        - actual_horizons
    )

    if missing_horizons:

        print(
            "WARNING: Missing common horizons: "
            f"{sorted(missing_horizons)}"
        )

    return comparison


# ============================================================================
# ADD MODEL RANKINGS
# ============================================================================


def add_model_rankings(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add NMAE ranking and best-model information.

    Rank 1 = lowest NMAE = best.
    """

    result = comparison.copy()

    nmae_columns = {
        "Persistence":
            "nmae_percent_persistence",

        "Linear Regression":
            "nmae_percent_linear_regression",

        "Feature-Engineered Linear Regression":
            "nmae_percent_feature_engineered_linear_regression",

        "Direct Multi-Horizon Linear Regression":
            "nmae_percent_direct_multi_horizon_linear_regression",
    }

    # ------------------------------------------------------------
    # Verify all expected columns exist.
    # ------------------------------------------------------------

    missing_columns = [
        column
        for column in nmae_columns.values()
        if column not in result.columns
    ]

    if missing_columns:

        raise KeyError(
            "Expected model NMAE columns are missing: "
            f"{missing_columns}"
        )

    # ------------------------------------------------------------
    # Find best model for every horizon.
    # ------------------------------------------------------------

    result["best_model"] = (
        result[
            list(nmae_columns.values())
        ]
        .idxmin(axis=1)
        .map(
            {
                column: model
                for model, column
                in nmae_columns.items()
            }
        )
    )

    result["best_nmae_percent"] = (
        result[
            list(nmae_columns.values())
        ]
        .min(axis=1)
    )

    # ------------------------------------------------------------
    # Add rank columns.
    # ------------------------------------------------------------

    ranking = (
        result[
            list(nmae_columns.values())
        ]
        .rank(
            axis=1,
            method="min",
            ascending=True,
        )
    )

    for model, column in nmae_columns.items():

        safe_name = (
            model.lower()
            .replace("-", "")
            .replace(" ", "_")
        )

        result[
            f"rank_{safe_name}"
        ] = ranking[column]

    return result


# ============================================================================
# BUILD DISPLAY TABLE
# ============================================================================


def build_display_table(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create the concise terminal report.
    """

    display = comparison[
        [
            "horizon",
            "nmae_percent_persistence",
            "nmae_percent_linear_regression",
            "nmae_percent_feature_engineered_linear_regression",
            "nmae_percent_direct_multi_horizon_linear_regression",
            "best_model",
            "best_nmae_percent",
        ]
    ].copy()

    display = display.rename(
        columns={
            "nmae_percent_persistence":
                "Persistence",

            "nmae_percent_linear_regression":
                "Linear Regression",

            "nmae_percent_feature_engineered_linear_regression":
                "Feature-Engineered LR",

            "nmae_percent_direct_multi_horizon_linear_regression":
                "Direct Multi-Horizon LR",

            "best_model":
                "Best Model",

            "best_nmae_percent":
                "Best NMAE (%)",
        }
    )

    return display


# ============================================================================
# SUMMARY
# ============================================================================


def print_summary(
    comparison: pd.DataFrame,
) -> None:
    """
    Print overall model performance summary.
    """

    model_columns = {
        "Persistence":
            "nmae_percent_persistence",

        "Linear Regression":
            "nmae_percent_linear_regression",

        "Feature-Engineered Linear Regression":
            "nmae_percent_feature_engineered_linear_regression",

        "Direct Multi-Horizon Linear Regression":
            "nmae_percent_direct_multi_horizon_linear_regression",
    }

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    # ------------------------------------------------------------
    # Count horizon wins.
    # ------------------------------------------------------------

    win_counts = {}

    for model, column in model_columns.items():

        win_counts[model] = int(
            (
                comparison["best_model"]
                == model
            ).sum()
        )

    print("HORIZON WINS")
    print("-" * 80)

    for model, wins in win_counts.items():

        print(
            f"{model}: {wins}/"
            f"{len(comparison)}"
        )

    # ------------------------------------------------------------
    # Overall mean NMAE.
    # ------------------------------------------------------------

    print()
    print("MEAN NMAE ACROSS TESTED HORIZONS")
    print("-" * 80)

    mean_scores = {}

    for model, column in model_columns.items():

        mean_scores[model] = (
            comparison[column].mean()
        )

    sorted_scores = sorted(
        mean_scores.items(),
        key=lambda item: item[1],
    )

    for rank, (model, score) in enumerate(
        sorted_scores,
        start=1,
    ):

        print(
            f"{rank}. {model}: "
            f"{score:.4f}%"
        )

    # ------------------------------------------------------------
    # Overall winner.
    # ------------------------------------------------------------

    overall_winner = (
        sorted_scores[0][0]
    )

    overall_score = (
        sorted_scores[0][1]
    )

    print()
    print(
        f"Overall best model: "
        f"{overall_winner}"
    )

    print(
        f"Mean NMAE: "
        f"{overall_score:.4f}%"
    )

    # ------------------------------------------------------------
    # Direct comparison: feature-engineered vs direct.
    # ------------------------------------------------------------

    fe_column = (
        "nmae_percent_"
        "feature_engineered_linear_regression"
    )

    direct_column = (
        "nmae_percent_"
        "direct_multi_horizon_linear_regression"
    )

    if (
        fe_column in comparison.columns
        and direct_column in comparison.columns
    ):

        direct_wins = int(
            (
                comparison[direct_column]
                < comparison[fe_column]
            ).sum()
        )

        feature_wins = int(
            (
                comparison[fe_column]
                < comparison[direct_column]
            ).sum()
        )

        ties = (
            len(comparison)
            - direct_wins
            - feature_wins
        )

        print()
        print(
            "DIRECT MULTI-HORIZON vs "
            "FEATURE-ENGINEERED LR"
        )
        print("-" * 80)

        print(
            f"Direct Multi-Horizon LR better: "
            f"{direct_wins}/{len(comparison)}"
        )

        print(
            f"Feature-Engineered LR better: "
            f"{feature_wins}/{len(comparison)}"
        )

        print(
            f"Ties: {ties}/{len(comparison)}"
        )


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:

    print("=" * 80)
    print("EIRGRID FORECAST MODEL COMPARISON")
    print("=" * 80)
    print()

    print(
        "Primary metric: NMAE (%)"
    )

    print(
        "Lower NMAE is better."
    )

    print()

    # ------------------------------------------------------------
    # Load all four benchmarks.
    # ------------------------------------------------------------

    benchmarks = {}

    for model_key, config in MODEL_CONFIG.items():

        benchmarks[model_key] = load_benchmark(
            path=config["file"],
            model_key=model_key,
            model_label=config["label"],
        )

    print()

    # ------------------------------------------------------------
    # Build comparison.
    # ------------------------------------------------------------

    comparison = build_comparison_table(
        benchmarks
    )

    # ------------------------------------------------------------
    # Add rankings.
    # ------------------------------------------------------------

    comparison = add_model_rankings(
        comparison
    )

    # ------------------------------------------------------------
    # Terminal display.
    # ------------------------------------------------------------

    display = build_display_table(
        comparison
    )

    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print()

    print(
        display.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------
    # Summary.
    # ------------------------------------------------------------

    print_summary(
        comparison
    )

    # ------------------------------------------------------------
    # Save complete comparison.
    # ------------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 80)
    print("OUTPUT")
    print("=" * 80)
    print()

    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print()
    print(
        "Benchmark comparison complete."
    )


if __name__ == "__main__":
    main()