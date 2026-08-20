"""
Persistence wind-power forecasting baseline.

The persistence forecast assumes that the next wind-generation value
is equal to the most recently observed value:

    P_forecast(t+1) = P_observed(t)

This is the baseline against which the later ML forecasting model
will be evaluated.

All power values are expressed in MW.
"""

from typing import List, Sequence


def persistence_forecast(
    wind_power_mw: Sequence[float],
    horizon: int = 1,
) -> List[float]:
    """
    Generate a persistence forecast.

    Parameters
    ----------
    wind_power_mw:
        Historical/observed wind-generation values in MW.

    horizon:
        Number of future periods to forecast.

    Returns
    -------
    List[float]
        Forecast wind generation for each future period.

    Examples
    --------
    >>> persistence_forecast([100.0, 120.0, 150.0])
    [150.0]

    >>> persistence_forecast([100.0, 120.0, 150.0], horizon=3)
    [150.0, 150.0, 150.0]
    """

    if len(wind_power_mw) == 0:
        raise ValueError(
            "wind_power_mw cannot be empty."
        )

    if horizon < 1:
        raise ValueError(
            "horizon must be at least 1."
        )

    last_value = float(
        wind_power_mw[-1]
    )

    if last_value < 0:
        raise ValueError(
            "Wind generation cannot be negative."
        )

    return [
        last_value
        for _ in range(horizon)
    ]