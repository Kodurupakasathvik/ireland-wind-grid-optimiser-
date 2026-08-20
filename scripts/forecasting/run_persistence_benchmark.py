"""
Run chronological persistence forecasting benchmark
on the real EirGrid 2026 wind-generation dataset.

Forecast horizons:
    15 min
    30 min
    1 hour
    2 hours
    4 hours

The benchmark uses only past observations to produce forecasts
and compares them with the actual future EirGrid measurements.
"""

from pathlib import Path

import pandas as pd

from scripts.data.wind_data_loader import load_wind_data
from scripts.forecasting.persistence_forecast import (
    persistence_forecast,
)
from scripts.forecasting.forecast_evaluation import (
    forecast_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "persistence_forecast_benchmark.csv"
)


HORIZONS = {
    "15min": 1,
    "30min": 2,
    "1hour": 4,
    "2hour": 8,
    "4hour": 16,
}


def run_benchmark() -> pd.DataFrame:

    dataframe = load_wind_data()

    wind = dataframe[
        "wind_generation_mw"
    ].dropna()

    rows = []

    for horizon_name, horizon_steps in HORIZONS.items():

        actual_values = []
        forecast_values = []

        # ------------------------------------------------------
        # Walk forward chronologically.
        #
        # At time t:
        #   forecast = wind(t)
        #
        # Target:
        #   wind(t + horizon)
        # ------------------------------------------------------

        for index in range(
            len(wind) - horizon_steps
        ):

            observed = float(
                wind.iloc[index]
            )

            actual = float(
                wind.iloc[
                    index + horizon_steps
                ]
            )

            forecast = persistence_forecast(
                [observed],
                horizon=1,
            )[0]

            actual_values.append(
                actual
            )

            forecast_values.append(
                forecast
            )

        metrics = forecast_metrics(
            actual_values,
            forecast_values,
        )

        rows.append(
            {
                "horizon": horizon_name,
                "horizon_steps": horizon_steps,
                "samples": len(actual_values),
                **metrics,
            }
        )

    results = pd.DataFrame(rows)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    return results


def main():

    print("=" * 80)
    print("EIRGRID PERSISTENCE FORECAST BENCHMARK")
    print("=" * 80)

    results = run_benchmark()

    print()
    print(results.to_string(index=False))

    print()
    print(
        f"Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()