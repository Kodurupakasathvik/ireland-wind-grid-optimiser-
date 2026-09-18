"""
Tests for the AC validation interface.
"""

from scripts.validation.ac_validator import (
    ACValidationResult,
    ACValidator,
    _reported_pf_convergence,
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


def test_existing_nonconvergence_diagnostic_message_is_preserved():
    """The release table should retain a concrete solver diagnostic."""

    result = ACValidationResult(
        converged=False,
        message="AC power flow produced nonphysical voltage magnitudes.",
    )

    validated = ACValidator().validate_result(result)

    assert validated.physically_secure is False
    assert "nonphysical voltage" in validated.message.lower()


def test_missing_or_nonfinite_core_metric_is_rejected():
    """Security claims fail closed when AC telemetry is incomplete."""

    validator = ACValidator()

    no_voltage = validator.validate_result(
        ACValidationResult(
            converged=True,
            maximum_line_loading_percent=80.0,
        )
    )
    nonfinite_loading = validator.validate_result(
        ACValidationResult(
            converged=True,
            minimum_voltage_pu=0.98,
            maximum_line_loading_percent=float("nan"),
        )
    )

    assert no_voltage.physically_secure is False
    assert "voltage" in no_voltage.message.lower()
    assert nonfinite_loading.physically_secure is False
    assert "line loading" in nonfinite_loading.message.lower()


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


def test_transformer_overload_is_rejected():
    """Thermal security covers transformers as well as lines."""

    validator = ACValidator()

    result = ACValidationResult(
        converged=True,
        minimum_voltage_pu=0.99,
        maximum_line_loading_percent=75.0,
        maximum_transformer_loading_percent=101.0,
        overloaded_transformers=["transformer_1"],
    )

    validated = validator.validate_result(result)

    assert validated.physically_secure is False
    assert "transformer" in validated.message.lower()


def test_explicit_pf_failure_is_not_inferred_from_result_arrays():
    """An explicit solver failure must override finite numeric result arrays."""

    class Report:
        columns = ["converged"]

        def __getitem__(self, key):
            assert key == "converged"
            return [False]

    assert _reported_pf_convergence({"main": Report()}) is False
