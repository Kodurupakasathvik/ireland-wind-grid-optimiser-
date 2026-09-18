# Ireland Wind Grid Optimiser

A transparent research prototype for studying wind dispatch-down in a stylised Irish transmission-network representation. It combines quarter-hourly EirGrid observations, lag-based time-series wind forecasts, a linear wind-acceptance optimiser, and nonlinear AC security screening.

This is a decision-support research model. It is **not** an EirGrid digital twin, operational dispatch tool, network-planning recommendation, or evidence of what a system operator would have done in a real event.

## Research question

How much observed wind dispatch-down could a forecast-informed optimiser reduce in a stylised Irish network, after each candidate dispatch is screened against configured nonlinear AC voltage and thermal criteria?

## Research workflow

```text
Quarter-hourly EirGrid observations
          |
          +-- Forecasting evaluation
          |      +-- horizon-specific wind forecasts
          |
          +-- Operating-scenario selection
                 +-- observed wind and demand snapshots
                              |
                              v
                 Linear wind-acceptance optimisation
                              |
                              v
                    LP potential dispatch result
                              |
                              v
                Nonlinear AC security screening
                       |                 |
                       v                 v
             security-screened      diagnostic /
                counterfactual      intervention analysis
```

The linear programme produces a candidate dispatch using linearised network constraints. The subsequent PyPSA AC power-flow calculation is a validation screen, not a full AC optimal-power-flow formulation. A linear-programme result remains **LP potential** until it passes the AC screen.

## Scope

The release contains:

- A stylised 58-bus, 65-line, 5-transformer transmission-network representation.
- Six selected independent 15-minute operating snapshots.
- Quarter-hourly EirGrid system and renewable-data observations.
- Persistence, linear-regression, feature-engineered, and direct multi-horizon wind-forecasting experiments.
- A linear optimisation model that maximises accepted wind subject to its configured constraints.
- Nonlinear AC validation of voltage, line loading, and transformer loading.
- A security-gated counterfactual calculation for energy, indicative value, and indicative emissions.

The network is intentionally a research representation. It does not reproduce confidential topology, operator control schemes, reserve policies, protection settings, outages, dynamic stability, or market dispatch arrangements.

## Terminology and interpretation

The project distinguishes four concepts that should not be conflated:

| Term | Meaning |
|---|---|
| Observed wind dispatch-down | The public-data quantity represented by observed wind not generated. |
| LP curtailment | Wind rejected by the linear optimisation model. |
| LP potential | The unscreened improvement suggested by the linear model. |
| Security-screened counterfactual | Energy eligible for interpretation only after the nonlinear AC screen passes. |

Each counterfactual row represents one independent 15-minute snapshot. Do not sum MW across snapshots. Energy may be aggregated only after multiplying by the 0.25-hour interval length and applying the AC-security gate.

## Forecasting methodology

The forecasting experiments use 20,348 quarter-hourly observations from 1 January to 31 July 2026.

Forecast issue time is the end of observation `t`; a forecast target at horizon `h` is at `t + h`. Lagged and rolling features are shifted so that they use only information available at `t` or earlier. The target is always later than the final feature observation.

Validation uses five expanding chronological windows:

- Initial training fraction: 60%.
- Test fraction per fold: 5%.
- No random shuffling.
- No future observations in lagged or rolling explanatory features.

These are retrospective forecasts based on recorded observations. The repository does not contain archived weather forecasts or operational message-receipt timestamps. It therefore does **not** claim measured forecast performance against the forecasts available to EirGrid dispatchers in real time.

See [`docs/FORECASTING_PROTOCOL.md`](docs/FORECASTING_PROTOCOL.md) for the full forecasting protocol.

## Optimisation and AC validation

The optimisation model maximises accepted wind subject to the configured linear constraints. Its result is then applied to the stylised network and checked using nonlinear AC power flow.

A candidate result is credited only when all of the following hold:

1. The AC calculation converges to a physically plausible numerical solution.
2. The configured minimum-voltage criterion is met.
3. Configured line-loading and transformer-loading criteria are met.
4. Required AC metrics are finite and available.

The screen fails closed: a non-converged, nonphysical, incomplete, or insecure AC result receives no energy, value, or emissions credit.

## Security-gated counterfactual results

[`data/processed/counterfactual_analysis.csv`](data/processed/counterfactual_analysis.csv) is the authoritative release output.

The base case has the following status:

| Scenario | AC status | Security-screened energy |
|---|---|---:|
| S1 - Normal | Converged but AC-insecure | 0 MWh |
| S2 - Peak demand | Not converged / non-creditable | 0 MWh |
| S3 - High wind | Not converged / non-creditable | 0 MWh |
| S4 - High wind, high demand | Not converged / non-creditable | 0 MWh |
| S5 - High availability, low generation | AC-security passed | 729.2375 MWh |
| S6 - Maximum stress | AC-security passed | 165.7325 MWh |
| **Selected-snapshot aggregate** | **Security-screened only** | **894.97 MWh** |

The 894.97 MWh aggregate is exactly the sum of the two AC-secure scenario credits. It is a selected-snapshot aggregate, **not** an annual saving, annual avoided-curtailment estimate, or forecast of real market outcomes.

At the base assumptions of EUR50/MWh and 0.4 tCO2/MWh, the eligible illustrative sensitivities are:

- Indicative value: **EUR44,748.50**
- Indicative emissions sensitivity: **357.988 tCO2**

These are assumption-driven counterfactual sensitivities. They are not historical market revenues, avoided-emissions measurements, or evidence that the model would have changed actual operator dispatch.

## S2 diagnostic status

`S2_PEAK_DEMAND` is retained as a stress-test diagnostic, not direct evidence of a physical system collapse.

The originally stored scenario is under-specified for a reliable AC conclusion. Controlled reconstruction can produce a conditional AC solution only under altered reactive-power assumptions; that solution still fails the configured voltage and thermal criteria. S2 is therefore excluded from AC-secure benefit claims.

See [`docs/S2_ASSESSMENT.md`](docs/S2_ASSESSMENT.md).

## Reactive-support and reinforcement sensitivities

Reactive-support experiments are exploratory engineering sensitivities, not device-sizing or investment recommendations.

For the recorded voltage-support sensitivity, reactive support improved the originally identified weak-bus voltage but did not remove system-wide thermal overloads:

| Reactive support | System-wide minimum voltage | Identified weak-bus voltage | Maximum line loading | Overloaded lines |
|---:|---:|---:|---:|---:|
| 0 MVAr | 0.667 pu | 0.667 pu | 175.4% | 9 |
| 300 MVAr | 0.907 pu | 0.931 pu | 165.0% | 9 |
| 400 MVAr | 0.938 pu | 0.979 pu | 165.0% | 9 |
| 500 MVAr | 0.956 pu | 1.021 pu | 165.3% | 9 |

At 400 MVAr, the identified weak bus rises from 0.667 pu to 0.979 pu, but the system-wide minimum remains below 0.95 pu and nine lines remain overloaded. Reactive support alone therefore does not establish a globally secure operating point.

A later Stage-5 reinforcement consolidation contains a metric-definition inconsistency: it records a voltage-security status while also recording a 0.917 pu minimum voltage associated with a virtual bus. Until that data dictionary and screening treatment are clarified, those reinforcement results are retained as diagnostics and are not promoted as evidence of a fully secure intervention.

## Reproducibility

Use Python 3.12 in a clean environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest scripts tests
python scripts/reproducibility/run_synthetic_fixture.py
python scripts/reproducibility/verify_release_artifacts.py
# Add --require-source-inputs when all manifest inputs are available locally.
python scripts/release/generate_release_figures.py
```

The synthetic fixture uses no EirGrid input and exercises the forecast-to-optimisation-to-AC-gate software path.

The release-artifact verification checks:

- Manifest hashes for source inputs available locally.
- Required release files.
- AC-security classification consistency.
- Non-crediting of non-converged and insecure cases.
- Structural validity of generated SVG figures.

## Key outputs

- [`data/processed/counterfactual_analysis.csv`](data/processed/counterfactual_analysis.csv) - security-gated counterfactual results.
- [`docs/FORECASTING_PROTOCOL.md`](docs/FORECASTING_PROTOCOL.md) - forecasting data, timing, feature, and validation protocol.
- [`docs/S2_ASSESSMENT.md`](docs/S2_ASSESSMENT.md) - S2 diagnostic assessment.
- [`data_manifest.csv`](data_manifest.csv) - input provenance, hashes, and redistribution status.
- [`reports/figures/`](reports/figures/) - reproducible release figures.
- [`scripts/reproducibility/`](scripts/reproducibility/) - synthetic fixture and release verification.
- [`scripts/validation/ac_validator.py`](scripts/validation/ac_validator.py) - shared AC-security gate.

## Limitations

- The network is stylised and should not be interpreted as the operational Irish transmission system.
- The LP uses a linearised network formulation; AC power flow is a post-optimisation screen only.
- The AC screen covers configured steady-state voltage and thermal criteria, not dynamic stability, inertia, reserves, protection, outages, voltage-control schemes, or operator actions.
- Selected scenarios are independent snapshots and do not form a chronological simulation.
- Forecast validation is retrospective and cannot establish operational forecast availability.
- Monetary and CO2 results depend on explicit assumptions.
- Reactive-support and reinforcement experiments omit device location, capability curves, dynamics, cost, constructability, and planning criteria.

## Data, licence, and responsible reuse

The code and original documentation are released under the [MIT License](LICENSE). External data remain subject to their own licences and terms.

The supplied raw and processed inputs are documented in [`data_manifest.csv`](data_manifest.csv). Re-download EirGrid workbooks from the official source and confirm current redistribution terms before publishing a derivative dataset. Some supplied geospatial extracts lack recorded acquisition provenance and should not be redistributed as verified source data.

Official EirGrid system and renewable-data publications are available from the [EirGrid system and renewable data reports page](https://www.eirgrid.ie/grid/system-and-renewable-data-reports).
