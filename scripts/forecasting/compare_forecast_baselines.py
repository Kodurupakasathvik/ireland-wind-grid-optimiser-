"""
Compare persistence and linear-regression wind-forecast baselines.

Reads benchmark results from data/processed and produces a
side-by-side comparison showing whether Linear Regression
improves on the persistence baseline.

The generated comparison CSV remains in data/processed,
which is intentionally ignored by Git.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

PROCESSED = ROOT / "data" / "processed"

PERSISTENCE_FILE = (
    PROCESSED / "persistence_forecast_benchmark.csv"
)

LINEAR_REGRESSION_FILE = (
    PROCESSED / "linear_regression_forecast_benchmark.csv"
)

OUTPUT_FILE = (
    PROCESSED / "forecast_baseline_comparison.csv"
)


def load_benchmark(path: Path, model_name: str) -> pd.DataFrame:
    """Load and validate one benchmark result."""

    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark file not found: {path}"
        )

    dataframe = pd.read_csv(path)

    required_columns = {
        "horizon",
        "horizon_steps",
        "samples",
        "mae_mw",
        "rmse_mw",
        "nmae",
        "nmae_percent",
    }

    missing = required_columns.difference(
        dataframe.columns
    )

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: "
            f"{sorted(missing)}"
        )

    dataframe = dataframe.copy()

    dataframe["model"] = model_name

    return dataframe


def main() -> None:

    print("=" * 80)
    print("PERSISTENCE vs LINEAR REGRESSION FORECAST COMPARISON")
    print("=" * 80)
    print()

    print("Loading persistence benchmark...")

    persistence = load_benchmark(
        PERSISTENCE_FILE,
        "Persistence",
    )

    print(
        f"  Horizons: {len(persistence)}"
    )

    print("Loading linear regression benchmark...")

    linear_regression = load_benchmark(
        LINEAR_REGRESSION_FILE,
        "Linear Regression",
    )

    print(
        f"  Horizons: {len(linear_regression)}"
    )

    # -------------------------------------------------------------
    # Merge the two benchmark sets by forecast horizon.
    # -------------------------------------------------------------

    comparison = persistence.merge(
        linear_regression,
        on=["horizon", "horizon_steps"],
        suffixes=("_persistence", "_linear_regression"),
        how="inner",
    )

    if comparison.empty:
        raise ValueError(
            "No common forecast horizons were found."
        )

    # -------------------------------------------------------------
    # Calculate improvement.
    #
    # Positive improvement means Linear Regression is better.
    # -------------------------------------------------------------

    comparison["mae_improvement_mw"] = (
        comparison["mae_mw_persistence"]
        - comparison["mae_mw_linear_regression"]
    )

    comparison["rmse_improvement_mw"] = (
        comparison["rmse_mw_persistence"]
        - comparison["rmse_mw_linear_regression"]
    )

    comparison["nmae_improvement_percent"] = (
        comparison["nmae_percent_persistence"]
        - comparison["nmae_percent_linear_regression"]
    )

    comparison["nmae_relative_improvement_percent"] = (
        (
            comparison["nmae_percent_persistence"]
            - comparison["nmae_percent_linear_regression"]
        )
        / comparison["nmae_percent_persistence"]
        * 100.0
    )

    comparison["linear_regression_better"] = (
        comparison["nmae_percent_linear_regression"]
        < comparison["nmae_percent_persistence"]
    )

    # -------------------------------------------------------------
    # Keep the report focused.
    # -------------------------------------------------------------

    report = comparison[
        [
            "horizon",
            "horizon_steps",
            "samples_persistence",
            "mae_mw_persistence",
            "mae_mw_linear_regression",
            "rmse_mw_persistence",
            "rmse_mw_linear_regression",
            "nmae_percent_persistence",
            "nmae_percent_linear_regression",
            "nmae_improvement_percent",
            "nmae_relative_improvement_percent",
            "linear_regression_better",
        ]
    ].copy()

    # -------------------------------------------------------------
    # Display.
    # -------------------------------------------------------------

    print()
    print("=" * 80)
    print("BASELINE COMPARISON")
    print("=" * 80)
    print()

    display_columns = [
        "horizon",
        "mae_mw_persistence",
        "mae_mw_linear_regression",
        "nmae_percent_persistence",
        "nmae_percent_linear_regression",
        "nmae_relative_improvement_percent",
        "linear_regression_better",
    ]

    print(
        report[display_columns].to_string(
            index=False
        )
    )

    print()

    # -------------------------------------------------------------
    # Summary.
    # -------------------------------------------------------------

    better_count = int(
        report["linear_regression_better"].sum()
    )

    total_count = len(report)

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    print(
        f"Linear Regression better: "
        f"{better_count}/{total_count} horizons"
    )

    if better_count == total_count:
        print(
            "Conclusion: Linear Regression outperforms "
            "persistence at every tested horizon."
        )

    elif better_count == 0:
        print(
            "Conclusion: Persistence outperforms "
            "Linear Regression at every tested horizon."
        )

    else:
        print(
            "Conclusion: Linear Regression improves on "
            "persistence at some horizons but not all."
        )

    print()

    # -------------------------------------------------------------
    # Save.
    # -------------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()