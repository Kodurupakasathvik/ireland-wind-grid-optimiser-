"""Run a self-contained forecast -> LP -> AC-security-gate smoke test.

The fixture deliberately uses synthetic data and a three-bus stylised network.
It validates the software path without redistributing or depending on any
EirGrid workbook. It is not a power-system result.

Run from the repository root:
    python scripts/reproducibility/run_synthetic_fixture.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.forecasting.persistence_forecast import persistence_forecast
from scripts.optimisation.linear_model import LinearLine, LinearNetwork
from scripts.optimisation.optimiser_types import OptimiserInput
from scripts.optimisation.wind_curtailment_optimizer import WindCurtailmentOptimizer
from scripts.validation.ac_validator import ACValidationResult, ACValidator


FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_wind.csv"
OUTPUT = ROOT / "results" / "synthetic_fixture_result.csv"


def read_last_observation() -> float:
    with FIXTURE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise ValueError("Synthetic fixture requires at least two observations.")
    return float(rows[-1]["wind_generation_mw"])


def main() -> None:
    observed_wind_mw = read_last_observation()
    forecast_wind_mw = float(persistence_forecast([observed_wind_mw], horizon=1)[0])

    network = LinearNetwork(
        buses=["slack", "load", "wind"],
        lines=[
            LinearLine("slack_load", "slack", "load", 10.0, 100.0),
            LinearLine("load_wind", "load", "wind", 10.0, 80.0),
        ],
        reference_bus="slack",
    )
    optimiser_input = OptimiserInput(
        snapshot="SYNTHETIC_15MIN",
        demand_mw={"load": 70.0},
        available_wind_mw={"synthetic_wind": forecast_wind_mw},
        wind_capacity_mw={"synthetic_wind": 100.0},
        conventional_generation_mw={},
        lines={},
        scenario="synthetic_fixture",
    )
    result = WindCurtailmentOptimizer(network).solve(
        optimiser_input,
        wind_bus={"synthetic_wind": "wind"},
    )

    # This fixture validates the gate contract itself. Full AC validation of
    # historical scenarios is performed by validate_pypsa_dispatch instead.
    ac_result = ACValidator().validate_result(
        ACValidationResult(
            converged=True,
            minimum_voltage_pu=0.98,
            maximum_line_loading_percent=75.0,
            scenario="synthetic_fixture",
            snapshot="SYNTHETIC_15MIN",
        )
    )
    if result.status != "optimal" or not ac_result.physically_secure:
        raise RuntimeError("Synthetic end-to-end fixture did not pass.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "snapshot",
                "observed_wind_mw",
                "forecast_wind_mw",
                "accepted_wind_mw",
                "curtailment_mw",
                "optimiser_status",
                "ac_security_passed",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "snapshot": "SYNTHETIC_15MIN",
                "observed_wind_mw": observed_wind_mw,
                "forecast_wind_mw": forecast_wind_mw,
                "accepted_wind_mw": result.accepted_wind_total_mw,
                "curtailment_mw": result.curtailment_total_mw,
                "optimiser_status": result.status,
                "ac_security_passed": ac_result.physically_secure,
            }
        )
    print(f"Synthetic fixture passed. Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
