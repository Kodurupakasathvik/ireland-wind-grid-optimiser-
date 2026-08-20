"""
EirGrid Forecast Model Selector
================================

Selects the best wind-forecast model using rolling time-series validation.

Primary criterion:
    Mean rolling-validation NMAE (%)

Secondary criteria:
    Cross-window NMAE standard deviation
    Worst-case NMAE

Models considered:
    1. Persistence
    2. Linear Regression
    3. Feature-Engineered Linear Regression
    4. Direct Multi-Horizon Linear Regression

Important:
    A model is allowed to win model selection only if rolling-validation
    results are available for the required horizons.

Fixed benchmark results are retained for reference but cannot override
rolling-validation results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

PROCESSED = ROOT / "data" / "processed"

BENCHMARK_FILE = (
    PROCESSED / "forecast_model_comparison.csv"
)

PERSISTENCE_ROLLING_FILE = (
    PROCESSED / "persistence_rolling_validation_summary.csv"
)

LINEAR_ROLLING_FILE = (
    PROCESSED / "linear_regression_rolling_validation_summary.csv"
)

DIRECT_ROLLING_FILE = (
    PROCESSED / "direct_multi_horizon_rolling_validation_summary.csv"
)

OUTPUT_FILE = (
    PROCESSED / "forecast_model_selection.csv"
)

SUMMARY_FILE = (
    PROCESSED / "forecast_model_selection_summary.csv"
)


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

MODELS = [
    "Persistence",
    "Linear Regression",
    "Feature-Engineered LR",
    "Direct Multi-Horizon LR",
]

ROLLING_MODELS = [
    "Persistence",
    "Linear Regression",
    "Direct Multi-Horizon LR",
]

HORIZONS = [
    "15min",
    "30min",
    "1hour",
    "2hour",
    "4hour",
]


# =============================================================================
# BENCHMARK COLUMN MAPPING
# =============================================================================

BENCHMARK_COLUMN_CANDIDATES = {
    "Persistence": [
        "Persistence",
        "persistence",
        "nmae_percent_persistence",
        "persistence_nmae_percent",
    ],
    "Linear Regression": [
        "Linear Regression",
        "linear_regression",
        "nmae_percent_linear_regression",
        "linear_regression_nmae_percent",
    ],
    "Feature-Engineered LR": [
        "Feature-Engineered LR",
        "Feature Engineered LR",
        "feature_engineered_lr",
        "nmae_percent_feature_engineered_linear_regression",
        "feature_engineered_linear_regression_nmae_percent",
    ],
    "Direct Multi-Horizon LR": [
        "Direct Multi-Horizon LR",
        "Direct Multi-Horizon Linear Regression",
        "direct_multi_horizon_lr",
        "nmae_percent_direct_multi_horizon_linear_regression",
        "direct_multi_horizon_linear_regression_nmae_percent",
    ],
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def normalize_name(value: Any) -> str:
    """Normalize a column/model name for flexible matching."""

    return (
        str(value)
        .strip()
        .lower()
        .replace("-", " ")
        .replace("_", " ")
        .replace("/", " ")
        .replace("  ", " ")
    )


def find_matching_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """Find a dataframe column using normalized candidate names."""

    normalized_columns = {
        normalize_name(column): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        normalized = normalize_name(candidate)

        if normalized in normalized_columns:
            return normalized_columns[normalized]

    return None


def canonical_horizon(value: Any) -> str | None:
    """
    Convert horizon representations into the canonical project labels.
    """

    text = str(value).strip().lower()

    mapping = {
        "15min": "15min",
        "15 min": "15min",
        "15mins": "15min",
        "15 minutes": "15min",
        "30min": "30min",
        "30 min": "30min",
        "30mins": "30min",
        "30 minutes": "30min",
        "1hour": "1hour",
        "1 hour": "1hour",
        "60min": "1hour",
        "60 min": "1hour",
        "2hour": "2hour",
        "2 hour": "2hour",
        "120min": "2hour",
        "120 min": "2hour",
        "4hour": "4hour",
        "4 hour": "4hour",
        "240min": "4hour",
        "240 min": "4hour",
    }

    if text in mapping:
        return mapping[text]

    return None


# =============================================================================
# BENCHMARK LOADING
# =============================================================================

def load_benchmark() -> pd.DataFrame:
    """
    Load the fixed all-model benchmark comparison.

    Fixed benchmark results are informational only.
    """

    print("Loading all-model benchmark comparison...")

    if not BENCHMARK_FILE.exists():
        raise FileNotFoundError(
            "Benchmark comparison file not found:\n"
            f"  {BENCHMARK_FILE}"
        )

    dataframe = pd.read_csv(BENCHMARK_FILE)

    print()
    print("Detected benchmark model columns:")

    detected: dict[str, str] = {}

    for model in MODELS:

        column = find_matching_column(
            dataframe,
            BENCHMARK_COLUMN_CANDIDATES[model],
        )

        if column is not None:
            detected[model] = column

            print(
                f"  {model}: {column}"
            )

    print()
    print(
        f"  Benchmark rows: {len(dataframe)}"
    )

    if not detected:
        raise ValueError(
            "No compatible model columns were found in "
            f"{BENCHMARK_FILE}."
        )

    result = pd.DataFrame()

    for model, column in detected.items():
        result[model] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    result["horizon"] = extract_benchmark_horizons(
        dataframe
    )

    return result


def extract_benchmark_horizons(
    dataframe: pd.DataFrame,
) -> list[str | None]:

    candidates = [
        "horizon",
        "Horizon",
        "forecast_horizon",
        "Forecast Horizon",
    ]

    column = find_matching_column(
        dataframe,
        candidates,
    )

    if column is None:
        return [None] * len(dataframe)

    return [
        canonical_horizon(value)
        for value in dataframe[column]
    ]


# =============================================================================
# ROLLING VALIDATION LOADING
# =============================================================================

def load_rolling_summary(
    path: Path,
    model_name: str,
) -> pd.DataFrame | None:
    """
    Load one rolling-validation summary.

    Expected fields include:
        horizon
        mean_nmae_percent
        std_nmae_percent
        worst_nmae_percent
    """

    print(
        f"Loading rolling-validation results for "
        f"{model_name}..."
    )

    if not path.exists():

        print(
            "  WARNING: Rolling validation file not found:"
        )
        print(
            f"    {path}"
        )

        return None

    dataframe = pd.read_csv(path)

    if dataframe.empty:

        print(
            "  WARNING: Rolling validation file is empty:"
        )
        print(
            f"    {path}"
        )

        return None

    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    # -------------------------------------------------------------------------
    # Horizon
    # -------------------------------------------------------------------------

    horizon_column = find_matching_column(
        dataframe,
        [
            "horizon",
            "forecast_horizon",
            "Forecast Horizon",
        ],
    )

    if horizon_column is None:
        raise ValueError(
            f"{path.name} is missing a horizon column."
        )

    dataframe["horizon"] = [
        canonical_horizon(value)
        for value in dataframe[horizon_column]
    ]

    # -------------------------------------------------------------------------
    # Mean NMAE
    # -------------------------------------------------------------------------

    mean_column = find_matching_column(
        dataframe,
        [
            "mean_nmae_percent",
            "mean NMAE (%)",
            "mean_nmae",
            "nmae_percent",
        ],
    )

    if mean_column is None:

        raise ValueError(
            f"{path.name} is missing mean NMAE information."
        )

    dataframe["mean_nmae_percent"] = pd.to_numeric(
        dataframe[mean_column],
        errors="coerce",
    )

    # -------------------------------------------------------------------------
    # Standard deviation
    # -------------------------------------------------------------------------

    std_column = find_matching_column(
        dataframe,
        [
            "std_nmae_percent",
            "std NMAE (%)",
            "nmae_std_percent",
            "cross_window_std_nmae_percent",
        ],
    )

    if std_column is not None:

        dataframe["std_nmae_percent"] = pd.to_numeric(
            dataframe[std_column],
            errors="coerce",
        )

    else:

        dataframe["std_nmae_percent"] = np.nan

    # -------------------------------------------------------------------------
    # Worst NMAE
    # -------------------------------------------------------------------------

    worst_column = find_matching_column(
        dataframe,
        [
            "worst_nmae_percent",
            "worst NMAE (%)",
            "max_nmae_percent",
        ],
    )

    if worst_column is not None:

        dataframe["worst_nmae_percent"] = pd.to_numeric(
            dataframe[worst_column],
            errors="coerce",
        )

    else:

        dataframe["worst_nmae_percent"] = np.nan

    # -------------------------------------------------------------------------
    # Keep only valid horizons
    # -------------------------------------------------------------------------

    dataframe = dataframe[
        dataframe["horizon"].isin(HORIZONS)
    ].copy()

    dataframe = dataframe[
        dataframe["mean_nmae_percent"].notna()
    ].copy()

    print(
        f"  {model_name}: "
        f"{dataframe['horizon'].nunique()} horizons"
    )

    return dataframe[
        [
            "horizon",
            "mean_nmae_percent",
            "std_nmae_percent",
            "worst_nmae_percent",
        ]
    ].copy()


# =============================================================================
# MODEL VALIDATION STATUS
# =============================================================================

def print_validation_status(
    rolling_results: dict[str, pd.DataFrame | None],
    benchmark: pd.DataFrame,
) -> None:

    print()
    print("=" * 80)
    print("MODEL VALIDATION STATUS")
    print("=" * 80)
    print()

    benchmark_horizons = set(
        benchmark["horizon"].dropna()
    )

    for model in MODELS:

        rolling = rolling_results.get(model)

        if rolling is None:
            rolling_count = 0
        else:
            rolling_count = (
                rolling["horizon"]
                .nunique()
            )

        fixed_count = 0

        if model in benchmark.columns:
            fixed_count = (
                benchmark[model]
                .notna()
                .sum()
            )

        if rolling_count == len(HORIZONS):

            status = "READY FOR MODEL SELECTION"

        elif rolling_count > 0:

            status = "PARTIAL ROLLING VALIDATION"

        else:

            status = "BENCHMARK ONLY"

        print(f"{model}:")
        print(
            f"  Rolling validation: "
            f"{rolling_count}/{len(HORIZONS)} horizons"
        )
        print(
            f"  Fixed benchmark: "
            f"{min(fixed_count, len(HORIZONS))}/"
            f"{len(HORIZONS)} horizons"
        )
        print(
            f"  Status: {status}"
        )
        print()


# =============================================================================
# HORIZON SELECTION
# =============================================================================

def select_best_models(
    rolling_results: dict[str, pd.DataFrame | None],
) -> pd.DataFrame:

    rows: list[dict[str, Any]] = []

    for horizon in HORIZONS:

        candidates: list[dict[str, Any]] = []

        for model, dataframe in rolling_results.items():

            if dataframe is None:
                continue

            subset = dataframe[
                dataframe["horizon"] == horizon
            ].copy()

            if subset.empty:
                continue

            row = subset.iloc[0]

            mean_nmae = float(
                row["mean_nmae_percent"]
            )

            std_nmae = float(
                row["std_nmae_percent"]
            ) if pd.notna(
                row["std_nmae_percent"]
            ) else float("inf")

            worst_nmae = float(
                row["worst_nmae_percent"]
            ) if pd.notna(
                row["worst_nmae_percent"]
            ) else float("inf")

            candidates.append(
                {
                    "model": model,
                    "mean_nmae_percent": mean_nmae,
                    "std_nmae_percent": std_nmae,
                    "worst_nmae_percent": worst_nmae,
                }
            )

        if not candidates:

            rows.append(
                {
                    "horizon": horizon,
                    "best_model": None,
                    "best_mean_nmae_percent": np.nan,
                    "best_std_nmae_percent": np.nan,
                    "best_worst_nmae_percent": np.nan,
                    "models_with_rolling_validation": 0,
                }
            )

            continue

        # Primary criterion:
        # mean NMAE
        #
        # Secondary:
        # standard deviation
        #
        # Tertiary:
        # worst-case NMAE

        candidates.sort(
            key=lambda item: (
                item["mean_nmae_percent"],
                item["std_nmae_percent"],
                item["worst_nmae_percent"],
            )
        )

        best = candidates[0]

        rows.append(
            {
                "horizon": horizon,
                "best_model": best["model"],
                "best_mean_nmae_percent": (
                    best["mean_nmae_percent"]
                ),
                "best_std_nmae_percent": (
                    best["std_nmae_percent"]
                ),
                "best_worst_nmae_percent": (
                    best["worst_nmae_percent"]
                ),
                "models_with_rolling_validation": (
                    len(candidates)
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# OVERALL MODEL RANKING
# =============================================================================

def build_overall_ranking(
    rolling_results: dict[str, pd.DataFrame | None],
) -> pd.DataFrame:

    rows: list[dict[str, Any]] = []

    for model in ROLLING_MODELS:

        dataframe = rolling_results.get(model)

        if dataframe is None or dataframe.empty:
            continue

        valid = dataframe[
            dataframe["horizon"].isin(HORIZONS)
        ].copy()

        if valid.empty:
            continue

        horizons_evaluated = (
            valid["horizon"].nunique()
        )

        mean_nmae = float(
            valid["mean_nmae_percent"].mean()
        )

        mean_std = float(
            valid["std_nmae_percent"].mean()
        )

        worst_nmae = float(
            valid["worst_nmae_percent"].max()
        )

        coverage = (
            horizons_evaluated
            / len(HORIZONS)
            * 100.0
        )

        rows.append(
            {
                "model": model,
                "horizons_evaluated": (
                    horizons_evaluated
                ),
                "coverage": coverage,
                "mean_nmae_percent": mean_nmae,
                "mean_cross_window_std_percent": (
                    mean_std
                ),
                "worst_nmae_percent": worst_nmae,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "overall_rank",
                "model",
                "horizons_evaluated",
                "coverage",
                "mean_nmae_percent",
                "mean_cross_window_std_percent",
                "worst_nmae_percent",
            ]
        )

    dataframe = pd.DataFrame(rows)

    dataframe = dataframe.sort_values(
        by=[
            "mean_nmae_percent",
            "mean_cross_window_std_percent",
            "worst_nmae_percent",
        ],
        ascending=[
            True,
            True,
            True,
        ],
    ).reset_index(drop=True)

    dataframe.insert(
        0,
        "overall_rank",
        np.arange(
            1,
            len(dataframe) + 1,
        ),
    )

    return dataframe


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 80)
    print("EIRGRID FORECAST MODEL SELECTOR")
    print("=" * 80)
    print()
    print(
        "Primary criterion: rolling-validation mean NMAE (%)"
    )
    print(
        "Secondary criteria: cross-window stability "
        "and worst-case NMAE."
    )

    # -------------------------------------------------------------------------
    # Load benchmark
    # -------------------------------------------------------------------------

    benchmark = load_benchmark()

    # -------------------------------------------------------------------------
    # Load ALL rolling validation results
    # -------------------------------------------------------------------------

    print()

    persistence = load_rolling_summary(
        PERSISTENCE_ROLLING_FILE,
        "Persistence",
    )

    linear = load_rolling_summary(
        LINEAR_ROLLING_FILE,
        "Linear Regression",
    )

    direct = load_rolling_summary(
        DIRECT_ROLLING_FILE,
        "Direct Multi-Horizon LR",
    )

    # Feature-Engineered LR deliberately remains benchmark-only
    # until a dedicated rolling-validation script is created.

    rolling_results: dict[
        str,
        pd.DataFrame | None,
    ] = {
        "Persistence": persistence,
        "Linear Regression": linear,
        "Direct Multi-Horizon LR": direct,
    }

    # -------------------------------------------------------------------------
    # Validation status
    # -------------------------------------------------------------------------

    print_validation_status(
        rolling_results,
        benchmark,
    )

    # -------------------------------------------------------------------------
    # Horizon model selection
    # -------------------------------------------------------------------------

    horizon_selection = select_best_models(
        rolling_results
    )

    print()
    print("=" * 80)
    print("HORIZON MODEL SELECTION")
    print("=" * 80)
    print()

    print(
        horizon_selection.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # -------------------------------------------------------------------------
    # Overall ranking
    # -------------------------------------------------------------------------

    overall_ranking = build_overall_ranking(
        rolling_results
    )

    print()
    print("=" * 80)
    print("OVERALL MODEL RANKING")
    print("=" * 80)
    print()

    if overall_ranking.empty:

        print(
            "No rolling-validation models are available."
        )

        return

    print(
        overall_ranking.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # -------------------------------------------------------------------------
    # Final recommendation
    # -------------------------------------------------------------------------

    recommended_model = (
        overall_ranking.iloc[0]["model"]
    )

    recommended_row = (
        overall_ranking.iloc[0]
    )

    print()
    print("=" * 80)
    print("FINAL FORECAST MODEL RECOMMENDATION")
    print("=" * 80)
    print()

    print(
        f"Recommended model: {recommended_model}"
    )

    print(
        "Mean rolling-validation NMAE: "
        f"{recommended_row['mean_nmae_percent']:.4f}%"
    )

    print(
        "Mean cross-window NMAE std: "
        f"{recommended_row['mean_cross_window_std_percent']:.4f}%"
    )

    print(
        "Worst observed NMAE: "
        f"{recommended_row['worst_nmae_percent']:.4f}%"
    )

    print(
        "Horizon coverage: "
        f"{int(recommended_row['horizons_evaluated'])}/"
        f"{len(HORIZONS)}"
    )

    # -------------------------------------------------------------------------
    # Benchmark-only models
    # -------------------------------------------------------------------------

    benchmark_only = []

    for model in MODELS:

        if model not in rolling_results:
            benchmark_only.append(model)

        elif rolling_results[model] is None:
            benchmark_only.append(model)

    if benchmark_only:

        print()
        print("IMPORTANT:")
        print(
            "The following models do not yet have "
            "rolling-validation results:"
        )

        for model in benchmark_only:
            print(f"  - {model}")

        print()
        print(
            "Their fixed benchmark results are retained "
            "for reference, but they are NOT allowed to "
            "beat rolling-validated models."
        )

    # -------------------------------------------------------------------------
    # Save horizon selection
    # -------------------------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    horizon_selection.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        f"Saved horizon selection: "
        f"{OUTPUT_FILE}"
    )

    # -------------------------------------------------------------------------
    # Save overall ranking
    # -------------------------------------------------------------------------

    overall_ranking.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print(
        f"Saved overall ranking: "
        f"{SUMMARY_FILE}"
    )

    print()
    print("=" * 80)
    print("MODEL SELECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()