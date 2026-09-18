"""
AC validation interface for the Ireland Wind Grid Optimiser.

Purpose
-------
This module provides the validation layer between:

    optimiser
        ↓
    accepted wind dispatch
        ↓
    PyPSA AC power flow
        ↓
    physical security metrics

The AC model is NOT the optimisation engine.

The optimiser decides how much wind to accept.
This module checks whether that dispatch is physically secure.

For the MVP, the validator supports a controlled validation
interface. The existing Irish 58-bus PyPSA network will be
connected through a network-loading adapter later.
"""

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import numpy as np
import pypsa


# These are deliberately broad numerical sanity bounds, not operating limits.
# Values outside them indicate a failed/nonphysical Newton-Raphson iterate and
# must not be reported as an AC solution. Actual security is assessed later at
# the configured 0.95 pu and 100% limits.
MAX_PLAUSIBLE_VOLTAGE_PU = 2.0
MAX_PLAUSIBLE_LOADING_PERCENT = 10_000.0


class _ACPowerFlowNotConverged(RuntimeError):
    """Raised when PyPSA explicitly reports an unconverged AC solve."""


def _reported_pf_convergence(power_flow_result: object) -> Optional[bool]:
    """Read PyPSA's explicit convergence flags, when they are available.

    ``Network.pf()`` commonly returns one or more tables containing a
    ``converged`` column. Earlier code inferred convergence solely from the
    presence of result arrays, which can misclassify a failed Newton-Raphson
    iteration with very large but finite values as a successful solution.

    The return value is ``None`` only for an unfamiliar or legacy return
    shape; in that case the caller retains the finite-value checks for
    compatibility.
    """

    reported: list[bool] = []

    def collect(value: object) -> None:
        if isinstance(value, MappingABC):
            for nested in value.values():
                collect(nested)
            return

        columns = getattr(value, "columns", None)
        if columns is None or "converged" not in columns:
            return

        flags = value["converged"]
        for flag in np.asarray(flags).ravel():
            if isinstance(flag, str):
                reported.append(flag.strip().lower() in {"true", "1", "yes"})
            elif flag is None or (isinstance(flag, float) and np.isnan(flag)):
                reported.append(False)
            else:
                reported.append(bool(flag))

    collect(power_flow_result)
    return all(reported) if reported else None


@dataclass
class ACValidationResult:
    """
    Result returned by the AC validation layer.
    """

    # AC power-flow status
    converged: bool

    # Voltage security
    minimum_voltage_pu: Optional[float] = None
    weak_bus: Optional[str] = None
    weak_bus_voltage_pu: Optional[float] = None

    # Thermal security
    maximum_line_loading_percent: Optional[float] = None
    overloaded_lines: List[str] = field(
        default_factory=list
    )
    maximum_transformer_loading_percent: Optional[float] = None
    overloaded_transformers: List[str] = field(
        default_factory=list
    )

    # General information
    scenario: str = "existing"
    snapshot: Optional[str] = None

    # Human-readable diagnostic
    message: str = ""

    # Final physical-security status.
    #
    # This is assigned by ACValidator.validate_result().
    _physically_secure: bool = False

    @property
    def physically_secure(self) -> bool:
        """
        Return whether the AC solution passed all configured
        physical-security checks.

        This value is deliberately stored separately from
        'converged' because a converged AC power flow can still
        violate voltage or thermal limits.
        """

        return self._physically_secure


class ACValidator:
    """
    Validation interface for the existing PyPSA AC model.

    Parameters
    ----------
    voltage_min_pu:
        Minimum acceptable bus voltage.

    line_loading_limit_percent:
        Maximum acceptable line loading.
    """

    def __init__(
        self,
        voltage_min_pu: float = 0.95,
        line_loading_limit_percent: float = 100.0,
    ) -> None:

        if voltage_min_pu <= 0:
            raise ValueError(
                "voltage_min_pu must be positive."
            )

        if line_loading_limit_percent <= 0:
            raise ValueError(
                "line_loading_limit_percent must be positive."
            )

        self.voltage_min_pu = float(
            voltage_min_pu
        )

        self.line_loading_limit_percent = float(
            line_loading_limit_percent
        )

    def validate_result(
        self,
        result: ACValidationResult,
    ) -> ACValidationResult:
        """
        Validate an already-computed AC result.

        Validation order:

        1. AC convergence
        2. Minimum voltage
        3. Maximum line loading
        4. Maximum transformer loading
        5. Explicit overloaded-element lists

        The result's physically_secure property is updated
        according to these checks.
        """

        # --------------------------------------------------
        # 1. AC convergence
        # --------------------------------------------------

        if not result.converged:

            result._physically_secure = False

            if not result.message:
                result.message = (
                    "AC power flow did not converge."
                )

            return result

        # --------------------------------------------------
        # 2. Required voltage and line-security metrics
        #
        # A result must fail closed if the AC adapter has not
        # supplied these core checks. In particular, a NaN does
        # not compare below a threshold and must never slip
        # through as a secure result.
        # --------------------------------------------------

        if (
            result.minimum_voltage_pu is None
            or not np.isfinite(result.minimum_voltage_pu)
        ):

            result._physically_secure = False

            result.message = (
                "AC solution is not security-creditable: "
                "minimum voltage is unavailable or non-finite."
            )

            return result

        if (
            result.maximum_line_loading_percent is None
            or not np.isfinite(result.maximum_line_loading_percent)
        ):

            result._physically_secure = False

            result.message = (
                "AC solution is not security-creditable: "
                "maximum line loading is unavailable or non-finite."
            )

            return result

        # --------------------------------------------------
        # 3. Voltage check
        # --------------------------------------------------

        if (
            result.minimum_voltage_pu is not None
            and result.minimum_voltage_pu
            < self.voltage_min_pu
        ):

            result._physically_secure = False

            result.message = (
                "AC solution is not secure: "
                "minimum bus voltage is below "
                f"{self.voltage_min_pu:.3f} pu."
            )

            return result

        # --------------------------------------------------
        # 4. Maximum line loading check
        # --------------------------------------------------

        if (
            result.maximum_line_loading_percent
            is not None
            and result.maximum_line_loading_percent
            > self.line_loading_limit_percent
        ):

            result._physically_secure = False

            result.message = (
                "AC solution is not secure: "
                "maximum line loading exceeds "
                f"{self.line_loading_limit_percent:.1f}%."
            )

            return result

        # --------------------------------------------------
        # 5. Maximum transformer loading check
        # --------------------------------------------------

        if (
            result.maximum_transformer_loading_percent
            is not None
            and result.maximum_transformer_loading_percent
            > self.line_loading_limit_percent
        ):

            result._physically_secure = False

            result.message = (
                "AC solution is not secure: "
                "maximum transformer loading exceeds "
                f"{self.line_loading_limit_percent:.1f}%."
            )

            return result

        # --------------------------------------------------
        # 6. Explicit overloaded elements
        # --------------------------------------------------

        if result.overloaded_lines or result.overloaded_transformers:

            result._physically_secure = False

            result.message = (
                "AC solution is not secure: "
                "one or more transmission elements "
                "are overloaded."
            )

            return result

        # --------------------------------------------------
        # 7. All checks passed
        # --------------------------------------------------

        result._physically_secure = True

        result.message = (
            "AC solution passed the configured "
            "physical security checks."
        )

        return result


def _loading_percent(
    network: pypsa.Network,
    component: str,
    snapshot: str,
) -> "np.ndarray | object":
    """Return apparent-power loading for lines or transformers.

    AC thermal ratings are apparent-power ratings.  When PyPSA provides
    ``s0`` it is therefore preferred to active power ``p0``; the latter is
    only a backwards-compatible fallback for older result tables.
    """

    static = getattr(network, component)
    dynamic = getattr(network, f"{component}_t")

    if static.empty:
        return np.array([])

    flow_column = "s0" if "s0" in dynamic else "p0"
    flow = dynamic[flow_column].loc[snapshot].abs()
    nominal = static["s_nom"].replace(0.0, np.nan)
    return flow / nominal * 100.0


def validate_pypsa_dispatch(
    network_path: str | Path,
    *,
    snapshot: str,
    scenario: str,
    demand_mw: Mapping[str, float],
    accepted_wind_mw: Mapping[str, float],
    wind_capacity_mw: Mapping[str, float],
    balancing_generator: str = "eirgrid_balancing_generation",
    voltage_min_pu: float = 0.95,
    thermal_loading_limit_percent: float = 100.0,
) -> ACValidationResult:
    """Validate an accepted-wind dispatch with nonlinear PyPSA AC power flow.

    This is the single release-path adapter used by the production,
    scenario-comparison, and counterfactual pipelines.  It deliberately
    separates a *converged* AC power flow from a *physically secure* result:
    voltage, line loading, and transformer loading must all pass before a
    dispatch may receive a security credit.
    """

    try:
        network = pypsa.Network(str(network_path))
        network.set_snapshots([snapshot])

        for generator, capacity in wind_capacity_mw.items():
            network.generators.at[generator, "p_nom"] = float(capacity)

        for load_name, load_row in network.loads.iterrows():
            bus = load_row["bus"]
            network.loads_t.p_set.loc[snapshot, load_name] = float(
                demand_mw.get(bus, 0.0)
            )

        for generator, accepted_mw in accepted_wind_mw.items():
            network.generators_t.p_set.loc[
                snapshot, generator
            ] = float(accepted_mw)

        if balancing_generator not in network.generators.index:
            raise KeyError(
                "AC validation network is missing the balancing generator "
                f"'{balancing_generator}'."
            )

        network.generators.at[balancing_generator, "p_nom"] = 20_000.0
        network.generators.at[balancing_generator, "p_min_pu"] = -1.0
        network.generators.at[balancing_generator, "p_max_pu"] = 1.0
        network.generators_t.p_set.loc[
            snapshot, balancing_generator
        ] = 0.0
        network.generators.at[balancing_generator, "control"] = "Slack"

        pf_result = network.pf()
        reported_convergence = _reported_pf_convergence(pf_result)
        if reported_convergence is False:
            raise _ACPowerFlowNotConverged(
                "PyPSA reported that the nonlinear AC power flow did not "
                "converge."
            )

        voltage = network.buses_t.v_mag_pu.loc[snapshot]
        if voltage.empty or not np.isfinite(voltage).all():
            raise ValueError("AC power flow did not produce finite voltages.")

        if (
            (voltage <= 0.0).any()
            or (voltage > MAX_PLAUSIBLE_VOLTAGE_PU).any()
        ):
            raise _ACPowerFlowNotConverged(
                "AC power flow produced nonphysical voltage magnitudes; "
                "the numerical iterate is not a validated solution."
            )

        line_loading = _loading_percent(network, "lines", snapshot)
        transformer_loading = _loading_percent(
            network, "transformers", snapshot
        )

        if not np.isfinite(line_loading).all() or not np.isfinite(
            transformer_loading
        ).all():
            raise ValueError("AC power flow did not produce finite thermal flows.")

        if (
            (line_loading > MAX_PLAUSIBLE_LOADING_PERCENT).any()
            or (transformer_loading > MAX_PLAUSIBLE_LOADING_PERCENT).any()
        ):
            raise _ACPowerFlowNotConverged(
                "AC power flow produced nonphysical thermal loading; "
                "the numerical iterate is not a validated solution."
            )

        weak_bus = str(voltage.idxmin())
        minimum_voltage = float(voltage.min())
        maximum_line_loading = (
            float(line_loading.max()) if len(line_loading) else None
        )
        maximum_transformer_loading = (
            float(transformer_loading.max())
            if len(transformer_loading)
            else None
        )
        overloaded_lines = list(
            line_loading[
                line_loading > thermal_loading_limit_percent
            ].index
        )
        overloaded_transformers = list(
            transformer_loading[
                transformer_loading > thermal_loading_limit_percent
            ].index
        )

        result = ACValidationResult(
            converged=True,
            minimum_voltage_pu=minimum_voltage,
            weak_bus=weak_bus,
            weak_bus_voltage_pu=minimum_voltage,
            maximum_line_loading_percent=maximum_line_loading,
            overloaded_lines=overloaded_lines,
            maximum_transformer_loading_percent=maximum_transformer_loading,
            overloaded_transformers=overloaded_transformers,
            scenario=scenario,
            snapshot=snapshot,
        )

    except _ACPowerFlowNotConverged as exc:
        result = ACValidationResult(
            converged=False,
            scenario=scenario,
            snapshot=snapshot,
            message=str(exc),
        )

    except Exception as exc:
        result = ACValidationResult(
            converged=False,
            scenario=scenario,
            snapshot=snapshot,
            message=f"AC power-flow execution failed: {exc}",
        )

    validator = ACValidator(
        voltage_min_pu=voltage_min_pu,
        line_loading_limit_percent=thermal_loading_limit_percent,
    )
    return validator.validate_result(result)
