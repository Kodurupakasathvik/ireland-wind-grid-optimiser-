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

from dataclasses import dataclass, field
from typing import List, Optional


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
        4. Explicit overloaded-line list

        The result's physically_secure property is updated
        according to these checks.
        """

        # --------------------------------------------------
        # 1. AC convergence
        # --------------------------------------------------

        if not result.converged:

            result._physically_secure = False

            result.message = (
                "AC power flow did not converge."
            )

            return result

        # --------------------------------------------------
        # 2. Voltage check
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
        # 3. Maximum line loading check
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
        # 4. Explicit overloaded lines
        # --------------------------------------------------

        if result.overloaded_lines:

            result._physically_secure = False

            result.message = (
                "AC solution is not secure: "
                "one or more transmission lines "
                "are overloaded."
            )

            return result

        # --------------------------------------------------
        # 5. All checks passed
        # --------------------------------------------------

        result._physically_secure = True

        result.message = (
            "AC solution passed the configured "
            "physical security checks."
        )

        return result