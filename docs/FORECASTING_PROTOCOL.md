# Forecasting protocol and validity statement

## Purpose

This document defines what the project’s wind forecasts measure and what
information is permitted at forecast issue time. It prevents retrospective
data leakage from being described as operational forecasting skill.

## Dataset and time convention

- Source workbook: `data/raw/System-Data-Qtr-Hourly-2026-V7.xlsx`.
- Series used: `IE Wind Generation` (with availability retained for the
  dispatch-down analysis).
- Frequency: 15 minutes.
- Observations retained by the current audit: 20,348, from 2026-01-01 00:00
  to 2026-07-31 23:45, with no duplicate timestamps.
- Timestamp convention: a row at time `t` is treated as information available
  when a forecast is issued at the end of interval `t`. This is an explicit
  research assumption; the workbook does not provide publication-latency or
  message-receipt timestamps.

## Forecast contract

For a forecast made at `t` with horizon `h` 15-minute steps:

```text
permitted features: observed wind and derived calendar/rolling features at t or earlier
prediction target:   wind generation at t + h
```

The feature code shifts historical series before calculating rolling
statistics. Therefore the target value is not included in its own predictors.
No random train/test shuffle is used. Missing observations are removed rather
than interpolated; the exact filtering is logged by the runner.

The horizons are 15 minutes, 30 minutes, 1 hour, 2 hours, and 4 hours. A
persistence model is included as the mandatory benchmark. The model selector
uses only rolling-validation summaries, not a fitted-in-sample score.

## Validation protocol

The direct multi-horizon and comparator experiments use five expanding,
chronological windows:

- initial train fraction: 60%;
- each test window: 5% of the cleaned series;
- training grows after each window; and
- every horizon is evaluated with a separately fitted model.

The selected model is chosen per horizon on mean rolling-validation NMAE. The
saved current selection is:

| Horizon | Selected model | Mean NMAE |
| --- | --- | ---: |
| 15 min | Direct Multi-Horizon LR | 2.44% |
| 30 min | Linear Regression | 3.66% |
| 1 hour | Linear Regression | 5.37% |
| 2 hours | Linear Regression | 8.47% |
| 4 hours | Linear Regression | 13.90% |

Full fold-level and summary outputs live in `data/processed/*rolling_validation*.csv`
and `data/processed/forecast_model_selection.csv`.

## What can and cannot be claimed

The evaluation supports this statement:

> On the supplied historical EirGrid observation series, the selected
> autoregressive models achieved the recorded expanding-window hindcast errors
> under the stated zero-latency information assumption.

It does **not** support a claim that the project reproduced or outperformed
operational EirGrid forecasts. That would require archived forecast issue
times, forecast values available at those issue times, and an evaluation that
preserves their delivery latency. Weather-station values are not used as
future-known forecast inputs.
