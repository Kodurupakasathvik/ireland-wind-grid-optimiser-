# =============================================================================
# S4.5F — TRANSFORMER + SLACK FORMULATION ISOLATION
# =============================================================================
#
# PURPOSE
# -------
# S4.5E demonstrated that:
#
#   Q = 0
#   + explicit slack
#   + distributed slack
#
# can produce finite voltages, but the physical/PF validation is still
# inconsistent.
#
# This stage isolates:
#
#   1. Transformer impedance representation
#   2. Transformer bypass sensitivity
#   3. Slack formulation
#   4. AC connected components
#   5. Voltage/angle stability
#
# NO P3 REINFORCEMENTS
# NO RESIDUAL REINFORCEMENTS
# NO REACTIVE SUPPORT
# NO SOURCE NETWORK MODIFICATION
#
# =============================================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pypsa


# =============================================================================
# CONFIGURATION
# =============================================================================

NETWORK_PATH = Path(
    "data/processed/eirgrid_second_reinforced_network.nc"
)

SNAPSHOT = "S2_PEAK_DEMAND"

OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = OUT_DIR / "s4_5f_transformer_slack_isolation.csv"
BUS_FILE = OUT_DIR / "s4_5f_bus_validation.csv"
LINE_FILE = OUT_DIR / "s4_5f_line_validation.csv"
TRAFO_FILE = OUT_DIR / "s4_5f_transformer_validation.csv"


# =============================================================================
# HEADER
# =============================================================================

print("=" * 100)
print("S4.5F — TRANSFORMER + SLACK FORMULATION ISOLATION")
print("=" * 100)

print(f"""
Network  : {NETWORK_PATH}
Snapshot : {SNAPSHOT}
PF       : AC nonlinear
Dispatch : unchanged
Loads    : unchanged
Reactive : temporarily Q=0
Source   : READ-ONLY

Cases:
  A — Q=0 + explicit slack + distributed slack + original transformers
  B — Q=0 + explicit slack + distributed slack + transformer impedances restored
  C — Q=0 + explicit slack ONLY + restored transformer impedances
  D — Q=0 + distributed slack + restored transformer impedances

No reinforcement is applied.
No reactive device is added.
No source network file is modified.
""")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def finite_series(series):
    """Return True if all values are finite."""
    arr = np.asarray(series, dtype=float)
    return np.isfinite(arr).all()


def get_snapshot_value(df, snapshot, column):
    """Safely extract one snapshot value."""
    if column not in df.columns:
        return np.nan

    if snapshot not in df.index:
        return np.nan

    return df.loc[snapshot, column]


def get_bus_voltage(n, snapshot):
    """Return bus voltage magnitude safely."""
    try:
        v = n.buses_t.v_mag_pu.loc[snapshot]
        v = pd.to_numeric(v, errors="coerce")
        return v
    except Exception:
        return pd.Series(dtype=float)


def get_bus_angle(n, snapshot):
    """Return bus angle safely."""
    try:
        a = n.buses_t.v_ang.loc[snapshot]
        a = pd.to_numeric(a, errors="coerce")
        return a
    except Exception:
        return pd.Series(dtype=float)


def safe_max_line_loading(n, snapshot):
    """
    Return maximum line loading percentage.

    PyPSA's loading can be derived from p0/s_nom and q0/s_nom.
    """
    try:
        if len(n.lines) == 0:
            return np.nan

        p0 = n.lines_t.p0.loc[snapshot].astype(float)
        q0 = n.lines_t.q0.loc[snapshot].astype(float)

        s0 = np.sqrt(p0**2 + q0**2)

        loading = 100.0 * s0 / n.lines.s_nom.astype(float)

        loading = loading.replace([np.inf, -np.inf], np.nan)

        if loading.dropna().empty:
            return np.nan

        return float(loading.max())

    except Exception:
        return np.nan


def count_overloaded_lines(n, snapshot):
    try:
        if len(n.lines) == 0:
            return np.nan

        p0 = n.lines_t.p0.loc[snapshot].astype(float)
        q0 = n.lines_t.q0.loc[snapshot].astype(float)

        s0 = np.sqrt(p0**2 + q0**2)

        loading = 100.0 * s0 / n.lines.s_nom.astype(float)

        loading = loading.replace([np.inf, -np.inf], np.nan)

        return int((loading > 100.0).sum())

    except Exception:
        return np.nan


def safe_max_transformer_loading(n, snapshot):
    try:
        if len(n.transformers) == 0:
            return np.nan

        p0 = n.transformers_t.p0.loc[snapshot].astype(float)
        q0 = n.transformers_t.q0.loc[snapshot].astype(float)

        s0 = np.sqrt(p0**2 + q0**2)

        loading = 100.0 * s0 / n.transformers.s_nom.astype(float)

        loading = loading.replace([np.inf, -np.inf], np.nan)

        if loading.dropna().empty:
            return np.nan

        return float(loading.max())

    except Exception:
        return np.nan


def get_pf_status(result):
    """
    Extract PyPSA PF result robustly.
    """
    converged = False
    error = np.nan
    iterations = np.nan

    try:
        conv = result.get("converged", None)

        if conv is not None:
            if isinstance(conv, pd.DataFrame):
                converged = bool(conv.loc[SNAPSHOT].all())
            elif isinstance(conv, pd.Series):
                converged = bool(conv.loc[SNAPSHOT])
            else:
                converged = bool(conv)
    except Exception:
        pass

    try:
        err = result.get("error", None)

        if err is not None:
            if isinstance(err, pd.DataFrame):
                error = float(err.loc[SNAPSHOT].max())
            elif isinstance(err, pd.Series):
                error = float(err.loc[SNAPSHOT])
            else:
                error = float(err)
    except Exception:
        pass

    try:
        nit = result.get("n_iter", None)

        if nit is not None:
            if isinstance(nit, pd.DataFrame):
                iterations = float(nit.loc[SNAPSHOT].max())
            elif isinstance(nit, pd.Series):
                iterations = float(nit.loc[SNAPSHOT])
            else:
                iterations = float(nit)
    except Exception:
        pass

    return converged, error, iterations


# =============================================================================
# TRANSFORMER SNAPSHOT
# =============================================================================

print("=" * 100)
print("LOADING SOURCE NETWORK")
print("=" * 100)

n_source = pypsa.Network(str(NETWORK_PATH))

print(f"Buses        : {len(n_source.buses)}")
print(f"Lines        : {len(n_source.lines)}")
print(f"Transformers : {len(n_source.transformers)}")
print(f"Generators   : {len(n_source.generators)}")
print(f"Loads        : {len(n_source.loads)}")


# =============================================================================
# SNAPSHOT ISOLATION
# =============================================================================

if SNAPSHOT not in n_source.snapshots:
    raise ValueError(
        f"Snapshot {SNAPSHOT!r} not found. "
        f"Available snapshots: {list(n_source.snapshots)}"
    )

n_source.set_snapshots([SNAPSHOT])

print()
print("=" * 100)
print("SNAPSHOT ISOLATION")
print("=" * 100)

print("Active snapshot:")
print(f"  {SNAPSHOT}")


# =============================================================================
# ORIGINAL TRANSFORMER DATA
# =============================================================================

print()
print("=" * 100)
print("ORIGINAL TRANSFORMER PARAMETERS")
print("=" * 100)

trafo_cols = [
    "bus0",
    "bus1",
    "s_nom",
    "r",
    "x",
    "tap_ratio",
    "phase_shift",
]

available_cols = [
    c for c in trafo_cols
    if c in n_source.transformers.columns
]

print(n_source.transformers[available_cols].to_string())


# =============================================================================
# SAVE ORIGINAL TRANSFORMER PARAMETERS IN MEMORY
# =============================================================================

original_r = n_source.transformers["r"].copy()
original_x = n_source.transformers["x"].copy()
original_tap = n_source.transformers["tap_ratio"].copy()
original_shift = n_source.transformers["phase_shift"].copy()


# =============================================================================
# TRANSFORMER CONDITIONING
# =============================================================================

print()
print("=" * 100)
print("TRANSFORMER CONDITIONING")
print("=" * 100)

for name, row in n_source.transformers.iterrows():

    r = float(row["r"])
    x = float(row["x"])

    print(
        f"{name:<45} "
        f"r={r:.12g}  "
        f"x={x:.12g}  "
        f"r/x="
        f"{(r/x if abs(x) > 1e-15 else np.inf):.6g}"
    )


# =============================================================================
# CASE FUNCTION
# =============================================================================

def run_case(
    case_name,
    transformer_mode="original",
    explicit_slack=True,
    distributed_slack=True,
):
    """
    Run one isolated baseline formulation.
    """

    print()
    print("=" * 100)
    print(f"CASE {case_name}")
    print("=" * 100)

    n = pypsa.Network(str(NETWORK_PATH))
    n.set_snapshots([SNAPSHOT])

    # -------------------------------------------------------------------------
    # Q = 0
    # -------------------------------------------------------------------------

    n.generators_t.q_set.loc[SNAPSHOT, :] = 0.0
    n.loads_t.q_set.loc[SNAPSHOT, :] = 0.0

    # -------------------------------------------------------------------------
    # Transformer handling
    # -------------------------------------------------------------------------

    if transformer_mode == "restored":

        print("Transformer mode : RESTORED ORIGINAL PARAMETERS")

        n.transformers["r"] = original_r.reindex(
            n.transformers.index
        ).fillna(n.transformers["r"])

        n.transformers["x"] = original_x.reindex(
            n.transformers.index
        ).fillna(n.transformers["x"])

        n.transformers["tap_ratio"] = original_tap.reindex(
            n.transformers.index
        ).fillna(n.transformers["tap_ratio"])

        n.transformers["phase_shift"] = original_shift.reindex(
            n.transformers.index
        ).fillna(n.transformers["phase_shift"])

    else:
        print("Transformer mode : SOURCE PARAMETERS")

    # -------------------------------------------------------------------------
    # Remove existing control assumptions
    # -------------------------------------------------------------------------

    n.generators["control"] = "PQ"

    # -------------------------------------------------------------------------
    # Clear previous slack assignment
    # -------------------------------------------------------------------------

    if "slack" in n.generators.columns:
        n.generators["control"] = "PQ"

    # -------------------------------------------------------------------------
    # Select positive dispatch generator
    # -------------------------------------------------------------------------

    try:
        p_dispatch = n.generators_t.p_set.loc[SNAPSHOT].astype(float)

        positive = p_dispatch[p_dispatch > 0]

    except Exception:
        positive = pd.Series(dtype=float)

    selected_slack = None

    if explicit_slack and not positive.empty:

        # Prefer the largest dispatch generator.
        selected_slack = positive.idxmax()

        n.generators.loc[selected_slack, "control"] = "Slack"

        print(f"Explicit slack generator : {selected_slack}")

    else:
        print("Explicit slack generator : NONE")

    # -------------------------------------------------------------------------
    # Distributed slack
    # -------------------------------------------------------------------------

    print(f"Distributed slack         : {distributed_slack}")

    # -------------------------------------------------------------------------
    # Run PF
    # -------------------------------------------------------------------------

    print()
    print("Running AC nonlinear power flow...")

    try:

        with warnings.catch_warnings(record=True) as caught:

            warnings.simplefilter("always")

            pf_result = n.pf(
                snapshots=[SNAPSHOT],
                distribute_slack=distributed_slack,
            )

        warning_count = len(caught)

    except Exception as exc:

        print("PF EXCEPTION:")
        print(repr(exc))

        return {
            "case": case_name,
            "transformer_mode": transformer_mode,
            "explicit_slack": explicit_slack,
            "distributed_slack": distributed_slack,
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
            "warning_count": warning_count if "warning_count" in locals() else np.nan,
            "slack_generator": selected_slack,
        }

    # -------------------------------------------------------------------------
    # PF status
    # -------------------------------------------------------------------------

    converged, pf_error, iterations = get_pf_status(pf_result)

    # -------------------------------------------------------------------------
    # Voltages
    # -------------------------------------------------------------------------

    v = get_bus_voltage(n, SNAPSHOT)

    finite_voltage = v.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if finite_voltage.empty:

        min_v = np.nan
        max_v = np.nan
        voltage_valid = False

    else:

        min_v = float(finite_voltage.min())
        max_v = float(finite_voltage.max())

        voltage_valid = (
            finite_voltage.between(0.5, 1.5).all()
        )

    # -------------------------------------------------------------------------
    # Angles
    # -------------------------------------------------------------------------

    a = get_bus_angle(n, SNAPSHOT)

    finite_angle = a.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if finite_angle.empty:
        min_angle = np.nan
        max_angle = np.nan
        angle_valid = False

    else:
        min_angle = float(finite_angle.min())
        max_angle = float(finite_angle.max())

        angle_valid = (
            finite_angle.abs().max() < np.pi
        )

    # -------------------------------------------------------------------------
    # Line loading
    # -------------------------------------------------------------------------

    max_line = safe_max_line_loading(n, SNAPSHOT)
    overloaded = count_overloaded_lines(n, SNAPSHOT)

    # -------------------------------------------------------------------------
    # Transformer loading
    # -------------------------------------------------------------------------

    max_trafo = safe_max_transformer_loading(n, SNAPSHOT)

    # -------------------------------------------------------------------------
    # Final physical validity
    # -------------------------------------------------------------------------

    valid_physical = (
        bool(converged)
        and voltage_valid
        and angle_valid
        and finite_series(v.values)
    )

    # -------------------------------------------------------------------------
    # Print
    # -------------------------------------------------------------------------

    print()
    print("-" * 100)
    print(f"CASE RESULT : {case_name}")
    print("-" * 100)

    print(f"Converged                  : {converged}")
    print(f"PF error                   : {pf_error}")
    print(f"Iterations                : {iterations}")
    print(f"Finite voltages            : {finite_series(v.values)}")
    print(f"Voltage range valid        : {voltage_valid}")
    print(f"Voltage minimum            : {min_v:.6f} pu")
    print(f"Voltage maximum            : {max_v:.6f} pu")
    print(f"Angle range valid          : {angle_valid}")
    print(f"Angle minimum              : {min_angle:.6f} rad")
    print(f"Angle maximum              : {max_angle:.6f} rad")
    print(f"Maximum line loading       : {max_line}")
    print(f"Overloaded lines           : {overloaded}")
    print(f"Maximum transformer load   : {max_trafo}")
    print(f"VALID PHYSICAL SOLUTION    : {valid_physical}")

    return {
        "case": case_name,
        "transformer_mode": transformer_mode,
        "explicit_slack": explicit_slack,
        "distributed_slack": distributed_slack,
        "converged": converged,
        "valid_physical_solution": valid_physical,
        "pf_error": pf_error,
        "iterations": iterations,
        "min_voltage_pu": min_v,
        "max_voltage_pu": max_v,
        "min_angle_rad": min_angle,
        "max_angle_rad": max_angle,
        "max_line_loading_pct": max_line,
        "overloaded_lines": overloaded,
        "max_transformer_loading_pct": max_trafo,
        "warning_count": warning_count,
        "slack_generator": selected_slack,
    }


# =============================================================================
# RUN CASES
# =============================================================================

results = []

results.append(
    run_case(
        case_name="A_Q0_ORIGINAL_TRAFO_DIST_SLACK",
        transformer_mode="original",
        explicit_slack=True,
        distributed_slack=True,
    )
)

results.append(
    run_case(
        case_name="B_Q0_RESTORED_TRAFO_DIST_SLACK",
        transformer_mode="restored",
        explicit_slack=True,
        distributed_slack=True,
    )
)

results.append(
    run_case(
        case_name="C_Q0_RESTORED_TRAFO_EXPLICIT_ONLY",
        transformer_mode="restored",
        explicit_slack=True,
        distributed_slack=False,
    )
)

results.append(
    run_case(
        case_name="D_Q0_RESTORED_TRAFO_DISTRIBUTED_ONLY",
        transformer_mode="restored",
        explicit_slack=False,
        distributed_slack=True,
    )
)


# =============================================================================
# SUMMARY
# =============================================================================

summary = pd.DataFrame(results)

print()
print("=" * 100)
print("S4.5F — SUMMARY")
print("=" * 100)

print(summary.to_string(index=False))


# =============================================================================
# SELECT BEST CASE
# =============================================================================

valid_cases = summary[
    summary["valid_physical_solution"] == True
]

print()
print("=" * 100)
print("VALIDATION RESULT")
print("=" * 100)

if valid_cases.empty:

    print("""
NO FULLY VALID CASE FOUND.

The investigation must continue into the network construction /
per-unit formulation before any voltage bottleneck conclusion
is accepted.
""")

else:

    print("VALID PHYSICAL CASE(S) FOUND:")

    print(
        valid_cases[
            [
                "case",
                "transformer_mode",
                "explicit_slack",
                "distributed_slack",
                "min_voltage_pu",
                "max_voltage_pu",
                "max_line_loading_pct",
                "overloaded_lines",
            ]
        ].to_string(index=False)
    )


# =============================================================================
# DETAILED TRANSFORMER VALIDATION
# =============================================================================

print()
print("=" * 100)
print("TRANSFORMER VALIDATION TABLE")
print("=" * 100)

trafo_validation = n_source.transformers[
    [
        "bus0",
        "bus1",
        "s_nom",
        "r",
        "x",
        "tap_ratio",
        "phase_shift",
    ]
].copy()

trafo_validation["r_finite"] = np.isfinite(
    trafo_validation["r"].astype(float)
)

trafo_validation["x_finite"] = np.isfinite(
    trafo_validation["x"].astype(float)
)

trafo_validation["zero_r"] = (
    trafo_validation["r"].astype(float).abs() < 1e-12
)

trafo_validation["zero_x"] = (
    trafo_validation["x"].astype(float).abs() < 1e-12
)

trafo_validation["r_over_x"] = np.where(
    trafo_validation["x"].abs() > 1e-12,
    trafo_validation["r"] / trafo_validation["x"],
    np.inf,
)

print(trafo_validation.to_string())


# =============================================================================
# BUS VALIDATION
# =============================================================================

print()
print("=" * 100)
print("BUS VALIDATION")
print("=" * 100)

bus_validation = n_source.buses[
    [
        "v_nom",
        "carrier",
    ]
].copy()

bus_validation["degree"] = 0

for bus in bus_validation.index:

    degree = 0

    if len(n_source.lines):

        degree += int(
            (
                (n_source.lines["bus0"] == bus)
                | (n_source.lines["bus1"] == bus)
            ).sum()
        )

    if len(n_source.transformers):

        degree += int(
            (
                (n_source.transformers["bus0"] == bus)
                | (n_source.transformers["bus1"] == bus)
            ).sum()
        )

    bus_validation.loc[bus, "degree"] = degree


print(bus_validation.to_string())


# =============================================================================
# SAVE
# =============================================================================

summary.to_csv(
    SUMMARY_FILE,
    index=False
)

bus_validation.to_csv(
    BUS_FILE
)

trafo_validation.to_csv(
    TRAFO_FILE
)

print()
print("=" * 100)
print("S4.5F RESULTS SAVED")
print("=" * 100)

print(f"Summary       : {SUMMARY_FILE}")
print(f"Bus validation: {BUS_FILE}")
print(f"Transformer   : {TRAFO_FILE}")

print()
print("=" * 100)
print("S4.5F COMPLETE")
print("=" * 100)

print("""
IMPORTANT:
  Source network modified : NO
  P3 reinforcement        : NO
  Residual reinforcement  : NO
  Reactive support        : NO
  Permanent changes       : NONE
""")