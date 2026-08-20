# scripts/s4_5h_q_continuation.py

import os
import warnings
import numpy as np
import pandas as pd
import pypsa

warnings.filterwarnings("ignore")

# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

NETWORK_PATH = r"data\processed\eirgrid_second_reinforced_network.nc"
SNAPSHOT = "S2_PEAK_DEMAND"

OUTPUT_SUMMARY = r"data\processed\s4_5h_q_continuation.csv"

Q_LEVELS = [0, 10, 25, 50, 75, 100]

LOAD_PF = 0.95

# Physical validation limits consistent with the previous diagnostic stages.
# The existing S4.5F/S4.5G reference solution has Vmin ~0.739 pu,
# therefore this stage must NOT impose an artificial 0.90 pu lower bound.
MIN_VALID_VOLTAGE = 0.0
MAX_VALID_VOLTAGE = 1.10

MAX_VALID_ANGLE_RAD = np.pi


# ==================================================================================================
# HEADER
# ==================================================================================================

print("=" * 100)
print("S4.5H — Q-CONTINUATION / REACTIVE POWER THRESHOLD ISOLATION — CORRECTED")
print("=" * 100)

print()
print(f"Network  : {NETWORK_PATH}")
print(f"Snapshot : {SNAPSHOT}")
print("PF       : AC nonlinear")
print("Dispatch : unchanged")
print("Loads P  : unchanged")
print("Generator Q : 0")
print("Slack    : distributed")
print("Source   : READ-ONLY")

print()
print("=" * 100)
print("PURPOSE")
print("=" * 100)

print("""
Determine the maximum load-reactive-power level that produces
a numerically converged and physically valid AC power-flow solution.

Continuation levels:
  0%   -> Q = 0 reference
  10%  -> 10% of realistic 0.95-PF load Q
  25%  -> 25%
  50%  -> 50%
  75%  -> 75%
  100% -> full 0.95-PF load Q

IMPORTANT:
  Every case is loaded independently from the source network.
  The S2_PEAK_DEMAND snapshot is explicitly selected.
  P dispatch is preserved.
  Generator Q is forced to zero.
  Load Q is scaled from the actual S2_PEAK_DEMAND P values.

No reinforcement is applied.
No reactive compensation is added.
No source network file is modified.
""")

# ==================================================================================================
# HELPER FUNCTIONS
# ==================================================================================================


def safe_float(value):
    """Convert a scalar-like value to float, returning NaN if impossible."""
    try:
        return float(value)
    except Exception:
        return np.nan


def physical_validity(
    converged,
    voltages,
    angles,
    line_loading,
    transformer_loading,
):
    """
    Validate the solved AC state.

    Convergence is mandatory.

    Voltage:
      finite
      positive
      <= 1.10 pu

    Angle:
      finite
      within +/- pi

    Loading:
      finite

    Overloads are NOT treated as numerical invalidity.
    They are reported separately because S4.5G itself produced
    a valid physical solution with 9 overloaded lines.
    """

    if not bool(converged):
        return False

    if voltages is None or len(voltages) == 0:
        return False

    if angles is None or len(angles) == 0:
        return False

    v = np.asarray(voltages, dtype=float)
    a = np.asarray(angles, dtype=float)

    if not np.all(np.isfinite(v)):
        return False

    if not np.all(np.isfinite(a)):
        return False

    if np.min(v) <= MIN_VALID_VOLTAGE:
        return False

    if np.max(v) > MAX_VALID_VOLTAGE:
        return False

    if np.max(np.abs(a)) > MAX_VALID_ANGLE_RAD:
        return False

    if line_loading is not None and len(line_loading) > 0:
        ll = np.asarray(line_loading, dtype=float)

        if not np.all(np.isfinite(ll)):
            return False

    if transformer_loading is not None and len(transformer_loading) > 0:
        tl = np.asarray(transformer_loading, dtype=float)

        if not np.all(np.isfinite(tl)):
            return False

    return True


def get_snapshot_series(df, snapshot):
    """
    Return one snapshot as a Series.

    This is the key protection against the previous
    'Length of values (8) does not match length of index (1)' error.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)

    if snapshot not in df.index:
        raise KeyError(
            f"Snapshot '{snapshot}' not found in time-dependent table."
        )

    return df.loc[snapshot].copy()


def calculate_realistic_load_q(p_values):
    """
    Calculate realistic lagging load Q from P at the specified PF.

        Q = P * tan(arccos(PF))
    """

    q_over_p = np.tan(np.arccos(LOAD_PF))

    p = pd.to_numeric(p_values, errors="coerce").fillna(0.0)

    return p * q_over_p


def run_case(q_percentage, source_path):
    """
    Run one completely isolated Q-continuation case.
    """

    case_name = f"Q_{int(q_percentage)}PCT"

    print()
    print("=" * 100)
    print(f"CASE {case_name}")
    print("=" * 100)

    print()
    print(f"Reactive load level : {q_percentage}%")
    print("Generator Q          : 0 Mvar")
    print("Distributed slack    : True")

    result = {
        "case": case_name,
        "q_percentage": float(q_percentage),
        "converged": False,
        "valid_physical_solution": False,
        "pf_error": np.nan,
        "iterations": np.nan,
        "min_voltage_pu": np.nan,
        "max_voltage_pu": np.nan,
        "min_angle_rad": np.nan,
        "max_angle_rad": np.nan,
        "max_line_loading_pct": np.nan,
        "overloaded_lines": np.nan,
        "max_transformer_loading_pct": np.nan,
        "overloaded_transformers": np.nan,
        "total_load_q_mvar": np.nan,
    }

    try:

        # ------------------------------------------------------------------------------------------
        # LOAD FRESH SOURCE NETWORK
        # ------------------------------------------------------------------------------------------

        n = pypsa.Network(source_path)

        # Explicitly isolate the requested snapshot.
        if SNAPSHOT not in n.snapshots:
            raise KeyError(
                f"Snapshot '{SNAPSHOT}' does not exist in network."
            )

        n.set_snapshots([SNAPSHOT])

        print()
        print("Fresh network loaded for this case.")
        print(f"Buses        : {len(n.buses)}")
        print(f"Lines        : {len(n.lines)}")
        print(f"Transformers : {len(n.transformers)}")
        print(f"Generators   : {len(n.generators)}")
        print(f"Loads        : {len(n.loads)}")

        # ------------------------------------------------------------------------------------------
        # READ ACTUAL S2 PEAK-DEMAND OPERATING POINT
        # ------------------------------------------------------------------------------------------

        p_gen = get_snapshot_series(
            n.generators_t.p_set,
            SNAPSHOT,
        )

        p_load = get_snapshot_series(
            n.loads_t.p_set,
            SNAPSHOT,
        )

        generator_p_total = float(
            pd.to_numeric(p_gen, errors="coerce")
            .fillna(0.0)
            .sum()
        )

        load_p_total = float(
            pd.to_numeric(p_load, errors="coerce")
            .fillna(0.0)
            .sum()
        )

        print()
        print("OPERATING POINT")
        print("-" * 100)
        print(f"Generator P set : {generator_p_total:.6f} MW")
        print(f"Load P set      : {load_p_total:.6f} MW")
        print(
            f"Generation-load : "
            f"{generator_p_total - load_p_total:.6f} MW"
        )

        # ------------------------------------------------------------------------------------------
        # REALISTIC REACTIVE LOAD REFERENCE
        # ------------------------------------------------------------------------------------------

        realistic_q = calculate_realistic_load_q(p_load)

        total_realistic_q = float(realistic_q.sum())

        scaled_q = realistic_q * (float(q_percentage) / 100.0)

        total_scaled_q = float(scaled_q.sum())

        result["total_load_q_mvar"] = total_scaled_q

        print()
        print("REACTIVE POWER")
        print("-" * 100)
        print(f"Load PF              : {LOAD_PF:.4f} lagging")
        print(
            f"Q/P ratio            : "
            f"{np.tan(np.arccos(LOAD_PF)):.6f}"
        )
        print(
            f"Full realistic Q     : "
            f"{total_realistic_q:.6f} Mvar"
        )
        print(
            f"Selected Q           : "
            f"{total_scaled_q:.6f} Mvar"
        )

        # ------------------------------------------------------------------------------------------
        # APPLY GENERATOR Q = 0
        # ------------------------------------------------------------------------------------------

        if not n.generators.empty:

            # Explicitly write ONE scalar to the requested snapshot.
            # This avoids assigning an 8-element array to a one-row index.
            n.generators_t.q_set.loc[
                SNAPSHOT,
                n.generators.index,
            ] = 0.0

        # ------------------------------------------------------------------------------------------
        # APPLY LOAD Q
        # ------------------------------------------------------------------------------------------

        if not n.loads.empty:

            # scaled_q is indexed by LOAD names, not snapshots.
            # Assign it directly into the single snapshot row.
            n.loads_t.q_set.loc[
                SNAPSHOT,
                scaled_q.index,
            ] = scaled_q.values

        # ------------------------------------------------------------------------------------------
        # VERIFY THE ASSIGNMENT BEFORE POWER FLOW
        # ------------------------------------------------------------------------------------------

        assigned_load_q = get_snapshot_series(
            n.loads_t.q_set,
            SNAPSHOT,
        )

        assigned_gen_q = get_snapshot_series(
            n.generators_t.q_set,
            SNAPSHOT,
        )

        assigned_load_q_total = float(
            pd.to_numeric(
                assigned_load_q,
                errors="coerce",
            )
            .fillna(0.0)
            .sum()
        )

        assigned_gen_q_total = float(
            pd.to_numeric(
                assigned_gen_q,
                errors="coerce",
            )
            .fillna(0.0)
            .sum()
        )

        print()
        print("Q ASSIGNMENT VERIFICATION")
        print("-" * 100)
        print(
            f"Generator Q assigned : "
            f"{assigned_gen_q_total:.6f} Mvar"
        )
        print(
            f"Load Q assigned      : "
            f"{assigned_load_q_total:.6f} Mvar"
        )

        # Hard safety check.
        if not np.isclose(
            assigned_load_q_total,
            total_scaled_q,
            rtol=1e-9,
            atol=1e-6,
        ):
            raise RuntimeError(
                "Load Q assignment verification failed."
            )

        if not np.isclose(
            assigned_gen_q_total,
            0.0,
            rtol=1e-9,
            atol=1e-6,
        ):
            raise RuntimeError(
                "Generator Q assignment verification failed."
            )

        # ------------------------------------------------------------------------------------------
        # DISTRIBUTED SLACK
        # ------------------------------------------------------------------------------------------

        # The previous successful S4.5G formulation used distributed slack
        # with eirgrid_non_wind_generation as the reference slack generator.
        #
        # We therefore preserve the generator control configuration from
        # the source network and explicitly identify the reference generator
        # for diagnostic reporting.

        slack_generator = "eirgrid_non_wind_generation"

        if slack_generator in n.generators.index:

            print()
            print("REFERENCE SLACK GENERATOR")
            print("-" * 100)
            print(f"Reference : {slack_generator}")

        else:

            print()
            print("WARNING")
            print("-" * 100)
            print(
                f"Expected reference generator "
                f"'{slack_generator}' was not found."
            )

        # ------------------------------------------------------------------------------------------
        # INITIAL VOLTAGE STATE
        # ------------------------------------------------------------------------------------------

        if not n.buses.empty:

            n.buses_t.v_mag_pu.loc[
                SNAPSHOT,
                n.buses.index,
            ] = 1.0

            n.buses_t.v_ang.loc[
                SNAPSHOT,
                n.buses.index,
            ] = 0.0

        # ------------------------------------------------------------------------------------------
        # RUN AC NONLINEAR POWER FLOW
        # ------------------------------------------------------------------------------------------

        print()
        print("Running AC nonlinear power flow...")

        pf_result = n.pf(
            snapshots=[SNAPSHOT],
            x_tol=1e-6,
            distribute_slack=True,
        )

        # ------------------------------------------------------------------------------------------
        # EXTRACT PF RESULT
        # ------------------------------------------------------------------------------------------

        converged_value = False
        pf_error = np.nan
        iterations = np.nan

        if isinstance(pf_result, dict):

            if "converged" in pf_result:

                conv_obj = pf_result["converged"]

                try:
                    converged_value = bool(
                        conv_obj.loc[SNAPSHOT].iloc[0]
                        if isinstance(conv_obj.loc[SNAPSHOT], pd.Series)
                        else conv_obj.loc[SNAPSHOT]
                    )
                except Exception:
                    try:
                        converged_value = bool(
                            np.asarray(conv_obj)[0]
                        )
                    except Exception:
                        converged_value = bool(conv_obj)

            if "error" in pf_result:

                err_obj = pf_result["error"]

                try:
                    value = err_obj.loc[SNAPSHOT]

                    if isinstance(value, pd.Series):
                        pf_error = safe_float(value.iloc[0])
                    else:
                        pf_error = safe_float(value)

                except Exception:
                    try:
                        pf_error = safe_float(
                            np.asarray(err_obj).reshape(-1)[0]
                        )
                    except Exception:
                        pass

            if "n_iter" in pf_result:

                iter_obj = pf_result["n_iter"]

                try:
                    value = iter_obj.loc[SNAPSHOT]

                    if isinstance(value, pd.Series):
                        iterations = safe_float(value.iloc[0])
                    else:
                        iterations = safe_float(value)

                except Exception:
                    try:
                        iterations = safe_float(
                            np.asarray(iter_obj).reshape(-1)[0]
                        )
                    except Exception:
                        pass

        # ------------------------------------------------------------------------------------------
        # VOLTAGES
        # ------------------------------------------------------------------------------------------

        voltages = pd.to_numeric(
            n.buses_t.v_mag_pu.loc[SNAPSHOT],
            errors="coerce",
        )

        angles = pd.to_numeric(
            n.buses_t.v_ang.loc[SNAPSHOT],
            errors="coerce",
        )

        finite_v = voltages[np.isfinite(voltages)]

        finite_a = angles[np.isfinite(angles)]

        # ------------------------------------------------------------------------------------------
        # LINE LOADING
        # ------------------------------------------------------------------------------------------

        line_loading = pd.Series(dtype=float)

        if not n.lines.empty:

            line_loading = pd.to_numeric(
                n.lines_t.p0.loc[SNAPSHOT].abs()
                / n.lines.s_nom.replace(0, np.nan)
                * 100.0,
                errors="coerce",
            )

        finite_line_loading = line_loading[
            np.isfinite(line_loading)
        ]

        overloaded_lines = int(
            (finite_line_loading > 100.0).sum()
        )

        # ------------------------------------------------------------------------------------------
        # TRANSFORMER LOADING
        # ------------------------------------------------------------------------------------------

        transformer_loading = pd.Series(dtype=float)

        if not n.transformers.empty:

            transformer_loading = pd.to_numeric(
                n.transformers_t.p0.loc[SNAPSHOT].abs()
                / n.transformers.s_nom.replace(0, np.nan)
                * 100.0,
                errors="coerce",
            )

        finite_transformer_loading = transformer_loading[
            np.isfinite(transformer_loading)
        ]

        overloaded_transformers = int(
            (finite_transformer_loading > 100.0).sum()
        )

        # ------------------------------------------------------------------------------------------
        # PHYSICAL VALIDATION
        # ------------------------------------------------------------------------------------------

        valid = physical_validity(
            converged=converged_value,
            voltages=finite_v,
            angles=finite_a,
            line_loading=finite_line_loading,
            transformer_loading=finite_transformer_loading,
        )

        # ------------------------------------------------------------------------------------------
        # STORE RESULTS
        # ------------------------------------------------------------------------------------------

        result["converged"] = converged_value
        result["valid_physical_solution"] = valid
        result["pf_error"] = pf_error
        result["iterations"] = iterations

        if len(finite_v) > 0:

            result["min_voltage_pu"] = float(finite_v.min())
            result["max_voltage_pu"] = float(finite_v.max())

        if len(finite_a) > 0:

            result["min_angle_rad"] = float(finite_a.min())
            result["max_angle_rad"] = float(finite_a.max())

        if len(finite_line_loading) > 0:

            result["max_line_loading_pct"] = float(
                finite_line_loading.max()
            )

        result["overloaded_lines"] = overloaded_lines

        if len(finite_transformer_loading) > 0:

            result["max_transformer_loading_pct"] = float(
                finite_transformer_loading.max()
            )

        result["overloaded_transformers"] = (
            overloaded_transformers
        )

        # ------------------------------------------------------------------------------------------
        # PRINT CASE RESULT
        # ------------------------------------------------------------------------------------------

        print()
        print("-" * 100)
        print(f"CASE RESULT : {case_name}")
        print("-" * 100)

        print(f"Converged                  : {converged_value}")
        print(f"PF error                   : {pf_error}")
        print(f"Iterations                : {iterations}")

        print(
            f"Finite voltages            : "
            f"{len(finite_v)}"
        )

        voltage_range_valid = (
            len(finite_v) > 0
            and np.min(finite_v) > MIN_VALID_VOLTAGE
            and np.max(finite_v) <= MAX_VALID_VOLTAGE
        )

        angle_range_valid = (
            len(finite_a) > 0
            and np.max(np.abs(finite_a)) <= MAX_VALID_ANGLE_RAD
        )

        print(
            f"Voltage range valid        : "
            f"{voltage_range_valid}"
        )

        if len(finite_v) > 0:

            print(
                f"Voltage minimum            : "
                f"{finite_v.min():.6f} pu"
            )

            print(
                f"Voltage maximum            : "
                f"{finite_v.max():.6f} pu"
            )

        print(
            f"Angle range valid          : "
            f"{angle_range_valid}"
        )

        if len(finite_a) > 0:

            print(
                f"Angle minimum              : "
                f"{finite_a.min():.6f} rad"
            )

            print(
                f"Angle maximum              : "
                f"{finite_a.max():.6f} rad"
            )

        if len(finite_line_loading) > 0:

            print(
                f"Maximum line loading       : "
                f"{finite_line_loading.max():.6f} %"
            )

        print(
            f"Overloaded lines           : "
            f"{overloaded_lines}"
        )

        if len(finite_transformer_loading) > 0:

            print(
                f"Maximum transformer load   : "
                f"{finite_transformer_loading.max():.6f} %"
            )

        print(
            f"Overloaded transformers    : "
            f"{overloaded_transformers}"
        )

        print(
            f"VALID PHYSICAL SOLUTION    : "
            f"{valid}"
        )

    except Exception as exc:

        print()
        print("CASE EXECUTION EXCEPTION")
        print(f"  {type(exc).__name__}: {exc}")

        result["converged"] = False
        result["valid_physical_solution"] = False

    return result


# ==================================================================================================
# SOURCE INSPECTION
# ==================================================================================================

print()
print("=" * 100)
print("LOADING SOURCE NETWORK")
print("=" * 100)

source_network = pypsa.Network(NETWORK_PATH)

if SNAPSHOT not in source_network.snapshots:
    raise RuntimeError(
        f"Required snapshot '{SNAPSHOT}' is not present."
    )

# Explicit snapshot isolation.
source_network.set_snapshots([SNAPSHOT])

print()
print("Snapshot successfully isolated:")
print(f"  {SNAPSHOT}")

# ------------------------------------------------------------------------------------------
# SOURCE P OPERATING POINT
# ------------------------------------------------------------------------------------------

source_generator_p = get_snapshot_series(
    source_network.generators_t.p_set,
    SNAPSHOT,
)

source_load_p = get_snapshot_series(
    source_network.loads_t.p_set,
    SNAPSHOT,
)

source_generator_p_total = float(
    pd.to_numeric(
        source_generator_p,
        errors="coerce",
    )
    .fillna(0.0)
    .sum()
)

source_load_p_total = float(
    pd.to_numeric(
        source_load_p,
        errors="coerce",
    )
    .fillna(0.0)
    .sum()
)

print()
print("=" * 100)
print("SOURCE OPERATING POINT")
print("=" * 100)

print(
    f"Generator P set : "
    f"{source_generator_p_total:.6f} MW"
)

print(
    f"Load P set      : "
    f"{source_load_p_total:.6f} MW"
)

print(
    f"Generation-load : "
    f"{source_generator_p_total - source_load_p_total:.6f} MW"
)

# ------------------------------------------------------------------------------------------
# REALISTIC Q REFERENCE
# ------------------------------------------------------------------------------------------

realistic_q_source = calculate_realistic_load_q(
    source_load_p
)

q_ratio = np.tan(np.arccos(LOAD_PF))

total_realistic_q = float(
    realistic_q_source.sum()
)

print()
print("=" * 100)
print("REALISTIC Q REFERENCE")
print("=" * 100)

print(
    f"Load PF              : "
    f"{LOAD_PF:.4f} lagging"
)

print(
    f"Q/P ratio            : "
    f"{q_ratio:.6f}"
)

print(
    f"Full realistic load Q : "
    f"{total_realistic_q:.6f} Mvar"
)

# ==================================================================================================
# RUN CONTINUATION
# ==================================================================================================

results = []

for q_percentage in Q_LEVELS:

    case_result = run_case(
        q_percentage=q_percentage,
        source_path=NETWORK_PATH,
    )

    results.append(case_result)


# ==================================================================================================
# SUMMARY
# ==================================================================================================

summary = pd.DataFrame(results)

# Ensure consistent column order.
summary = summary[
    [
        "case",
        "q_percentage",
        "converged",
        "valid_physical_solution",
        "pf_error",
        "iterations",
        "min_voltage_pu",
        "max_voltage_pu",
        "min_angle_rad",
        "max_angle_rad",
        "max_line_loading_pct",
        "overloaded_lines",
        "max_transformer_loading_pct",
        "overloaded_transformers",
        "total_load_q_mvar",
    ]
]

print()
print("=" * 100)
print("S4.5H — CONTINUATION SUMMARY")
print("=" * 100)

print(summary.to_string(index=False))


# ==================================================================================================
# THRESHOLD INTERPRETATION
# ==================================================================================================

valid_rows = summary[
    summary["valid_physical_solution"] == True
]

invalid_rows = summary[
    summary["valid_physical_solution"] == False
]

print()
print("=" * 100)
print("THRESHOLD INTERPRETATION")
print("=" * 100)

if len(valid_rows) > 0:

    highest_valid = valid_rows.iloc[-1]

    print(
        f"Highest physically valid tested Q level : "
        f"{highest_valid['q_percentage']:.0f}%"
    )

    # First invalid level after the highest valid level.
    later_invalid = summary[
        summary["q_percentage"]
        > highest_valid["q_percentage"]
    ]

    if len(later_invalid) > 0:

        first_invalid = later_invalid.iloc[0]

        print(
            f"First invalid tested Q level       : "
            f"{first_invalid['q_percentage']:.0f}%"
        )

        print(
            f"Observed transition                : "
            f"{highest_valid['q_percentage']:.0f}% -> "
            f"{first_invalid['q_percentage']:.0f}%"
        )

    else:

        print(
            "First invalid tested Q level       : NONE"
        )

        print(
            "Observed transition                : "
            "NOT OBSERVED WITHIN TESTED RANGE"
        )

else:

    print(
        "Highest physically valid tested Q level : NONE"
    )

    if len(invalid_rows) > 0:

        print(
            f"First invalid tested Q level       : "
            f"{invalid_rows.iloc[0]['q_percentage']:.0f}%"
        )

    print(
        "Observed transition                : "
        "NOT DETERMINED"
    )


# ==================================================================================================
# REFERENCE CHECK
# ==================================================================================================

print()
print("=" * 100)
print("Q=0 REFERENCE CONSISTENCY CHECK")
print("=" * 100)

q0 = summary[
    summary["q_percentage"] == 0
]

if len(q0) == 1:

    q0 = q0.iloc[0]

    print(
        f"Q=0 converged           : "
        f"{q0['converged']}"
    )

    print(
        f"Q=0 physically valid    : "
        f"{q0['valid_physical_solution']}"
    )

    print(
        f"Q=0 PF error            : "
        f"{q0['pf_error']}"
    )

    print(
        f"Q=0 iterations          : "
        f"{q0['iterations']}"
    )

    print(
        f"Q=0 minimum voltage     : "
        f"{q0['min_voltage_pu']:.6f} pu"
    )

    print(
        f"Q=0 maximum voltage     : "
        f"{q0['max_voltage_pu']:.6f} pu"
    )

    print(
        f"Q=0 overloaded lines    : "
        f"{q0['overloaded_lines']}"
    )

    # Expected S4.5G reference values.
    expected_vmin = 0.738929
    expected_vmax = 1.000326
    expected_overloaded_lines = 9

    reference_match = (
        bool(q0["converged"])
        and bool(q0["valid_physical_solution"])
        and np.isclose(
            q0["min_voltage_pu"],
            expected_vmin,
            atol=1e-3,
        )
        and np.isclose(
            q0["max_voltage_pu"],
            expected_vmax,
            atol=1e-3,
        )
        and int(q0["overloaded_lines"])
        == expected_overloaded_lines
    )

    print()
    print(
        f"Q=0 reproduces S4.5G reference : "
        f"{reference_match}"
    )

    if not reference_match:

        print()
        print("WARNING:")
        print(
            "The Q=0 continuation case does not reproduce "
            "the established S4.5G reference."
        )
        print(
            "Do NOT interpret the reactive-power threshold "
            "until this discrepancy is resolved."
        )

else:

    print("Q=0 result missing.")


# ==================================================================================================
# SAVE
# ==================================================================================================

os.makedirs(
    os.path.dirname(OUTPUT_SUMMARY),
    exist_ok=True,
)

summary.to_csv(
    OUTPUT_SUMMARY,
    index=False,
)

print()
print("=" * 100)
print("S4.5H RESULTS SAVED")
print("=" * 100)

print(
    f"Summary       : {OUTPUT_SUMMARY}"
)

print()
print("=" * 100)
print("S4.5H COMPLETE")
print("=" * 100)

print()
print("SOURCE NETWORK MODIFIED : NO")
print("REINFORCEMENTS APPLIED  : NO")
print("REACTIVE DEVICES ADDED  : NO")
print("PERMANENT CHANGES       : NONE")
print("=" * 100)