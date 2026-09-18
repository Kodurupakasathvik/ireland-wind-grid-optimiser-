"""Tests for the AC-security gate on counterfactual value claims."""

from scripts.pipeline.run_counterfactual_analysis import (
    security_screened_counterfactual_metrics,
)
from scripts.validation.ac_validator import ACValidationResult, ACValidator


def _ac_result(*, converged: bool, voltage: float | None, loading: float | None):
    result = ACValidationResult(
        converged=converged,
        minimum_voltage_pu=voltage,
        maximum_line_loading_percent=loading,
    )
    return ACValidator().validate_result(result)


def test_insecure_ac_result_receives_no_credit():
    """LP potential must never become a valued counterfactual when insecure."""

    metrics = security_screened_counterfactual_metrics(
        actual_not_generated_mw=100.0,
        optimised_curtailment_mw=20.0,
        ac_result=_ac_result(converged=True, voltage=0.94, loading=90.0),
        interval_hours=0.25,
    )

    assert metrics["lp_potential_curtailment_reduction_mw"] == 80.0
    assert metrics["security_screened_energy_mwh"] == 0.0
    assert metrics["eligible_for_value_credit"] is False
    assert metrics["security_assessment"] == "AC_INSECURE_NOT_CREDITED"


def test_non_converged_ac_result_receives_no_credit():
    """A failed AC solve is not interpreted as zero curtailment or a benefit."""

    metrics = security_screened_counterfactual_metrics(
        actual_not_generated_mw=100.0,
        optimised_curtailment_mw=20.0,
        ac_result=_ac_result(converged=False, voltage=None, loading=None),
        interval_hours=0.25,
    )

    assert metrics["security_screened_curtailment_reduction_mw"] == 0.0
    assert metrics["eligible_for_value_credit"] is False
    assert metrics["security_assessment"] == "AC_NOT_CONVERGED_NOT_CREDITED"


def test_secure_ac_result_can_receive_credit():
    """A value credit requires every configured AC check to pass."""

    metrics = security_screened_counterfactual_metrics(
        actual_not_generated_mw=100.0,
        optimised_curtailment_mw=20.0,
        ac_result=_ac_result(converged=True, voltage=0.98, loading=90.0),
        interval_hours=0.25,
    )

    assert metrics["security_screened_curtailment_reduction_mw"] == 80.0
    assert metrics["security_screened_energy_mwh"] == 20.0
    assert metrics["eligible_for_value_credit"] is True
