"""
Tests for the AC validation interface.
"""

from scripts.validation.ac_validator import (
    ACValidationResult,
    ACValidator,
)


def test_secure_ac_result_is_accepted():
    """
    A converged solution with acceptable voltage and
    thermal loading should be considered physically secure.
    """

    validator = ACValidator(
        voltage_min_pu=0.95,
        line_loading_limit_percent=100.0,
    )

    result = ACValidationResult(
        converged=True,
        minimum_voltage_pu=0.98,
        weak_bus="bus_2",
        weak_bus_voltage_pu=0.98,
        maximum_line_loading_percent=85.0,
        overloaded_lines=[],
        scenario="existing",
        snapshot="TEST",
    )

    validated = validator.validate_result(result)

    assert validated.converged is True
    assert validated.physically_secure is True
    assert "passed" in validated.message.lower()


def test_low_voltage_is_rejected():
    """
    A converged solution with voltage below the threshold
    should fail physical security validation.
    """

    validator = ACValidator(
        voltage_min_pu=0.95,
        line_loading_limit_percent=100.0,
    )

    result = ACValidationResult(
        converged=True,
        minimum_voltage_pu=0.92,
        weak_bus="bus_2",
        weak_bus_voltage_pu=0.92,
        maximum_line_loading_percent=80.0,
        overloaded_lines=[],
        scenario="existing",
        snapshot="TEST",
    )

    validated = validator.validate_result(result)

    assert validated.converged is True
    assert validated.physically_secure is False
    assert "voltage" in validated.message.lower()


def test_thermal_overload_is_rejected():
    """
    A converged solution with excessive line loading
    should fail physical security validation.
    """

    validator = ACValidator(
        voltage_min_pu=0.95,
        line_loading_limit_percent=100.0,
    )

    result = ACValidationResult(
        converged=True,
        minimum_voltage_pu=0.98,
        weak_bus="bus_2",
        weak_bus_voltage_pu=0.98,
        maximum_line_loading_percent=125.0,
        overloaded_lines=["line_1"],
        scenario="existing",
        snapshot="TEST",
    )

    validated = validator.validate_result(result)

    assert validated.converged is True
    assert validated.physically_secure is False
    assert "loading" in validated.message.lower()


def test_non_converged_ac_solution_is_rejected():
    """
    A non-converged AC power flow must always fail validation.
    """

    validator = ACValidator()

    result = ACValidationResult(
        converged=False,
        minimum_voltage_pu=None,
        maximum_line_loading_percent=None,
        overloaded_lines=[],
        scenario="existing",
        snapshot="TEST",
    )

    validated = validator.validate_result(result)

    assert validated.converged is False
    assert validated.physically_secure is False
    assert "converge" in validated.message.lower()


def test_zero_overloads_can_be_secure():
    """
    A converged solution with acceptable voltage and
    zero overloads should pass.
    """

    validator = ACValidator()

    result = ACValidationResult(
        converged=True,
        minimum_voltage_pu=1.00,
        weak_bus="bus_1",
        weak_bus_voltage_pu=1.00,
        maximum_line_loading_percent=50.0,
        overloaded_lines=[],
        scenario="existing",
        snapshot="S2_PEAK_DEMAND",
    )

    validated = validator.validate_result(result)

    assert validated.physically_secure is True