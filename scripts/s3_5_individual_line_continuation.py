# scripts/s3_5_individual_line_continuation.py
#
# S3.5 INDIVIDUAL-LINE REINFORCEMENT
# CONTROLLED AC CONTINUATION — BALANCE CORRECTED V5
#
# Purpose:
#   Test individual transmission-line reinforcement under a balanced
#   S2_PEAK_DEMAND operating point using nonlinear AC power flow.
#
# Key design:
#   1. Balance the original S2 system exactly.
#   2. Keep wind + interconnector generation fixed proportionally.
#   3. Balance remaining generation using the non-wind generator.
#   4. Apply individual line reinforcement only.
#   5. Use controlled continuation in lambda.
#   6. Preserve the last CONVERGED voltage state.
#   7. On failure, retry from the last valid state using smaller lambda steps.
#   8. Never use a diverged voltage state as the next initial condition.
#
# Network:
#   data/processed/eirgrid_optimized_network.nc
#
# Snapshot:
#   S2_PEAK_DEMAND
#
# Candidates:
#   merged_way/257889771-220+1
#   way/343436171-220
#   merged_way/1231251986-220+2
#   merged_way/61295764-220+1
#   merged_relation/4872159-220+1
#
# Reinforcements:
#   1.25x
#   1.50x
#
# Continuation target:
#   lambda = 1.0
#
# ----------------------------------------------------------------------

from pathlib import Path
import copy
import math
import warnings

import numpy as np
import pandas as pd
import pypsa


# ============================================================================
# CONFIGURATION
# ============================================================================

NETWORK_PATH = Path(
    "data/processed/eirgrid_optimized_network.nc"
)

SNAPSHOT = "S2_PEAK_DEMAND"

CANDIDATES = [
    "merged_way/257889771-220+1",
    "way/343436171-220",
    "merged_way/1231251986-220+2",
    "merged_way/61295764-220+1",
    "merged_relation/4872159-220+1",
]

REINFORCEMENTS = [
    1.25,
    1.50,
]

# Original coarse continuation points requested for S3.5
INITIAL_LAMBDAS = [
    0.900,
    0.925,
    0.950,
    0.975,
    1.000,
]

# ----------------------------------------------------------------------
# Adaptive continuation controls
# ----------------------------------------------------------------------

# Maximum lambda step.
MAX_STEP = 0.025

# Minimum lambda step before declaring continuation failure.
MIN_STEP = 0.001

# After a successful solve, step can increase toward MAX_STEP.
STEP_GROWTH = 1.5

# After failure, shrink the step.
STEP_SHRINK = 0.5

# Number of retries at a failed lambda.
MAX_RETRIES = 8

# Power-flow tolerance.
X_TOL = 1e-8

# Maximum voltage magnitude considered numerically valid.
MAX_VALID_V = 2.0

# Minimum voltage magnitude considered numerically valid.
MIN_VALID_V = 0.05

# Numerical sanity limit.
MAX_ABS_VALUE = 1e8

# We want exact balance before every PF.
BALANCE_TOL = 1e-7


# ============================================================================
# PRINTING
# ============================================================================

WIDTH = 110


def header(title):
    print()
    print("=" * WIDTH)
    print(title)
    print("=" * WIDTH)


def subheader(title):
    print()
    print("-" * WIDTH)
    print(title)
    print("-" * WIDTH)


def fmt(x):
    if x is None:
        return "N/A"
    if isinstance(x, float):
        return f"{x:.6f}"
    return str(x)


# ============================================================================
# NETWORK / DATA HELPERS
# ============================================================================

def load_network():
    if not NETWORK_PATH.exists():
        raise FileNotFoundError(
            f"Network not found:\n{NETWORK_PATH.resolve()}"
        )

    n = pypsa.Network(NETWORK_PATH)

    if SNAPSHOT not in n.snapshots:
        raise RuntimeError(
            f"Snapshot '{SNAPSHOT}' not found.\n"
            f"Available snapshots: {list(n.snapshots)}"
        )

    return n


def ensure_numeric(series, default=0.0):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def get_generator_groups(n):
    """
    Identify wind, interconnector and non-wind generators.

    Expected carriers in the current network:
        wind
        non_wind
        interconnector

    The function deliberately uses carrier names instead of relying
    on generator ordering.
    """

    if n.generators.empty:
        raise RuntimeError("Network contains no generators.")

    carriers = (
        n.generators["carrier"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    wind_mask = carriers.eq("wind")
    interconnector_mask = carriers.eq("interconnector")
    non_wind_mask = carriers.eq("non_wind")

    wind = list(n.generators.index[wind_mask])
    interconnectors = list(n.generators.index[interconnector_mask])
    non_wind = list(n.generators.index[non_wind_mask])

    if not wind:
        raise RuntimeError("No generators with carrier='wind' found.")

    if not interconnectors:
        raise RuntimeError(
            "No generators with carrier='interconnector' found."
        )

    if not non_wind:
        raise RuntimeError(
            "No generators with carrier='non_wind' found."
        )

    return wind, interconnectors, non_wind


# ============================================================================
# GENERATOR OUTPUT HELPERS
# ============================================================================

def get_generator_output(n, generator):
    """
    Return generator output for SNAPSHOT.

    Prefer p_set because the S2 reference is a controlled dispatch.
    Fall back to p if necessary.
    """

    if (
        hasattr(n, "generators_t")
        and hasattr(n.generators_t, "p_set")
        and generator in n.generators_t.p_set.columns
        and SNAPSHOT in n.generators_t.p_set.index
    ):
        value = n.generators_t.p_set.at[SNAPSHOT, generator]

        if pd.notna(value):
            return float(value)

    if (
        hasattr(n, "generators_t")
        and hasattr(n.generators_t, "p")
        and generator in n.generators_t.p.columns
        and SNAPSHOT in n.generators_t.p.index
    ):
        value = n.generators_t.p.at[SNAPSHOT, generator]

        if pd.notna(value):
            return float(value)

    raise RuntimeError(
        f"Cannot determine output for generator '{generator}'."
    )


def set_generator_output(n, generator, value):
    """
    Set both p_set and p where available.

    The important quantity for AC PF is p_set.
    """

    if generator not in n.generators.index:
        raise KeyError(generator)

    if SNAPSHOT not in n.generators_t.p_set.index:
        n.generators_t.p_set.loc[SNAPSHOT, generator] = value
    else:
        n.generators_t.p_set.at[SNAPSHOT, generator] = value

    # p is an output/result table. It is not normally necessary to set it,
    # but keeping it synchronized helps avoid stale values during diagnostics.
    if SNAPSHOT in n.generators_t.p.index:
        n.generators_t.p.at[SNAPSHOT, generator] = value


# ============================================================================
# POWER BALANCE
# ============================================================================

def calculate_original_balance(n):
    """
    Calculate the original S2 generation/load balance.

    This is diagnostic only.
    """

    load = float(
        n.loads_t.p_set.loc[SNAPSHOT].sum()
    )

    generation = float(
        n.generators_t.p_set.loc[SNAPSHOT].sum()
    )

    mismatch = generation - load

    return generation, load, mismatch


def build_balanced_reference(n):
    """
    Construct the balanced S2 reference.

    Current model logic:

        fixed generation =
            wind + interconnector

        balancing generation =
            non_wind

    The total system is then exactly balanced:

        wind
      + interconnector
      + non_wind
      = load

    We preserve the original wind and interconnector dispatch and
    calculate the required non-wind output.

    IMPORTANT:
    We do NOT multiply the original total generation and then try to
    compensate afterward. We construct the balance explicitly.
    """

    wind, interconnectors, non_wind = get_generator_groups(n)

    load = float(
        n.loads_t.p_set.loc[SNAPSHOT].sum()
    )

    wind_generation = sum(
        get_generator_output(n, g)
        for g in wind
    )

    interconnector_generation = sum(
        get_generator_output(n, g)
        for g in interconnectors
    )

    original_non_wind = sum(
        get_generator_output(n, g)
        for g in non_wind
    )

    fixed_generation = (
        wind_generation
        + interconnector_generation
    )

    required_non_wind = (
        load
        - fixed_generation
    )

    if required_non_wind < 0:
        raise RuntimeError(
            "Required non-wind generation is negative.\n"
            f"Load: {load:.6f} MW\n"
            f"Fixed generation: {fixed_generation:.6f} MW\n"
            f"Required non-wind: {required_non_wind:.6f} MW"
        )

    # ------------------------------------------------------------------
    # Dispatch non-wind generation.
    #
    # If there are multiple non-wind generators, preserve their original
    # proportions.
    # ------------------------------------------------------------------

    if len(non_wind) == 1:

        set_generator_output(
            n,
            non_wind[0],
            required_non_wind,
        )

    else:

        original_values = np.array(
            [
                get_generator_output(n, g)
                for g in non_wind
            ],
            dtype=float,
        )

        original_total = original_values.sum()

        if original_total <= 0:
            weights = np.ones(len(non_wind)) / len(non_wind)
        else:
            weights = (
                original_values
                / original_total
            )

        for g, weight in zip(non_wind, weights):
            set_generator_output(
                n,
                g,
                required_non_wind * weight,
            )

    # ------------------------------------------------------------------
    # Verify exact balance.
    # ------------------------------------------------------------------

    balanced_generation = float(
        n.generators_t.p_set.loc[SNAPSHOT].sum()
    )

    mismatch = (
        balanced_generation
        - load
    )

    if abs(mismatch) > BALANCE_TOL:

        # Correct the first non-wind generator by the residual.
        first_non_wind = non_wind[0]

        current = get_generator_output(
            n,
            first_non_wind,
        )

        set_generator_output(
            n,
            first_non_wind,
            current - mismatch,
        )

        balanced_generation = float(
            n.generators_t.p_set.loc[SNAPSHOT].sum()
        )

        mismatch = (
            balanced_generation
            - load
        )

    if abs(mismatch) > BALANCE_TOL:

        raise RuntimeError(
            "Unable to create balanced S2 reference.\n"
            f"Generation: {balanced_generation:.12f} MW\n"
            f"Load: {load:.12f} MW\n"
            f"Mismatch: {mismatch:.12e} MW"
        )

    return {
        "wind": wind,
        "interconnectors": interconnectors,
        "non_wind": non_wind,
        "load": load,
        "wind_generation": wind_generation,
        "interconnector_generation": interconnector_generation,
        "original_non_wind": original_non_wind,
        "fixed_generation": fixed_generation,
        "required_non_wind": required_non_wind,
    }


# ============================================================================
# APPLY LAMBDA
# ============================================================================

def apply_lambda(n, reference, lam):
    """
    Apply a continuation loading factor.

    Loads:
        lambda * original load

    Fixed generation:
        lambda * original wind
        lambda * original interconnector

    Balancing generation:
        lambda * original load
        - lambda * fixed generation

    Therefore:

        total generation == total load

    for every lambda.
    """

    original_load = reference["load"]
    original_fixed = reference["fixed_generation"]

    scaled_load = lam * original_load
    scaled_fixed = lam * original_fixed

    required_non_wind = (
        scaled_load
        - scaled_fixed
    )

    # ------------------------------------------------------------------
    # Scale all loads from their ORIGINAL reference values.
    # ------------------------------------------------------------------

    for load_name in n.loads.index:

        original_value = float(
            reference["load_values"][load_name]
        )

        n.loads_t.p_set.at[
            SNAPSHOT,
            load_name
        ] = lam * original_value

    # ------------------------------------------------------------------
    # Scale wind.
    # ------------------------------------------------------------------

    for g in reference["wind"]:

        original_value = float(
            reference["wind_values"][g]
        )

        set_generator_output(
            n,
            g,
            lam * original_value,
        )

    # ------------------------------------------------------------------
    # Scale interconnectors.
    # ------------------------------------------------------------------

    for g in reference["interconnectors"]:

        original_value = float(
            reference["interconnector_values"][g]
        )

        set_generator_output(
            n,
            g,
            lam * original_value,
        )

    # ------------------------------------------------------------------
    # Set non-wind generation.
    # ------------------------------------------------------------------

    non_wind = reference["non_wind"]

    if len(non_wind) == 1:

        set_generator_output(
            n,
            non_wind[0],
            required_non_wind,
        )

    else:

        weights = reference["non_wind_weights"]

        for g in non_wind:

            set_generator_output(
                n,
                g,
                required_non_wind * weights[g],
            )

    # ------------------------------------------------------------------
    # Final balance correction.
    # ------------------------------------------------------------------

    generation = float(
        n.generators_t.p_set.loc[SNAPSHOT].sum()
    )

    load = float(
        n.loads_t.p_set.loc[SNAPSHOT].sum()
    )

    mismatch = generation - load

    if abs(mismatch) > BALANCE_TOL:

        first_non_wind = non_wind[0]

        current = get_generator_output(
            n,
            first_non_wind,
        )

        set_generator_output(
            n,
            first_non_wind,
            current - mismatch,
        )

        generation = float(
            n.generators_t.p_set.loc[SNAPSHOT].sum()
        )

        load = float(
            n.loads_t.p_set.loc[SNAPSHOT].sum()
        )

        mismatch = generation - load

    if abs(mismatch) > BALANCE_TOL:

        raise RuntimeError(
            f"Lambda balance failure at {lam:.6f}.\n"
            f"Generation: {generation:.12f}\n"
            f"Load: {load:.12f}\n"
            f"Mismatch: {mismatch:.12e}"
        )

    return (
        generation,
        load,
        mismatch,
        lam * reference["fixed_generation"],
        required_non_wind,
    )


# ============================================================================
# REFERENCE CAPTURE
# ============================================================================

def capture_reference_values(n, reference):
    """
    Store immutable original dispatch/load values so every lambda
    is calculated from the SAME S2 reference.

    This prevents cumulative scaling errors.
    """

    reference["load_values"] = {
        load_name: float(
            n.loads_t.p_set.at[
                SNAPSHOT,
                load_name
            ]
        )
        for load_name in n.loads.index
    }

    reference["wind_values"] = {
        g: float(
            n.generators_t.p_set.at[
                SNAPSHOT,
                g
            ]
        )
        for g in reference["wind"]
    }

    reference["interconnector_values"] = {
        g: float(
            n.generators_t.p_set.at[
                SNAPSHOT,
                g
            ]
        )
        for g in reference["interconnectors"]
    }

    reference["non_wind_values"] = {
        g: float(
            n.generators_t.p_set.at[
                SNAPSHOT,
                g
            ]
        )
        for g in reference["non_wind"]
    }

    total_non_wind = sum(
        reference["non_wind_values"].values()
    )

    if total_non_wind <= 0:

        weight = 1.0 / len(reference["non_wind"])

        reference["non_wind_weights"] = {
            g: weight
            for g in reference["non_wind"]
        }

    else:

        reference["non_wind_weights"] = {
            g:
            reference["non_wind_values"][g]
            / total_non_wind
            for g in reference["non_wind"]
        }


# ============================================================================
# VOLTAGE STATE MANAGEMENT
# ============================================================================

def capture_voltage_state(n):
    """
    Capture ONLY the AC voltage state from a converged solution.

    This state is later used to seed the next continuation point.
    """

    state = {}

    if hasattr(n.buses_t, "v_mag_pu"):
        state["v_mag_pu"] = (
            n.buses_t.v_mag_pu
            .loc[[SNAPSHOT]]
            .copy(deep=True)
        )

    if hasattr(n.buses_t, "v_ang"):
        state["v_ang"] = (
            n.buses_t.v_ang
            .loc[[SNAPSHOT]]
            .copy(deep=True)
        )

    if hasattr(n.buses_t, "v_ang_pu"):
        state["v_ang_pu"] = (
            n.buses_t.v_ang_pu
            .loc[[SNAPSHOT]]
            .copy(deep=True)
        )

    return state


def restore_voltage_state(n, state):
    """
    Restore a previously converged voltage state.

    We deliberately restore ONLY the voltage variables.
    """

    if not state:
        return

    if (
        "v_mag_pu" in state
        and hasattr(n.buses_t, "v_mag_pu")
    ):
        source = state["v_mag_pu"]

        common = [
            x for x in source.columns
            if x in n.buses_t.v_mag_pu.columns
        ]

        if common:
            n.buses_t.v_mag_pu.loc[
                SNAPSHOT,
                common
            ] = source.loc[
                SNAPSHOT,
                common
            ]

    if (
        "v_ang" in state
        and hasattr(n.buses_t, "v_ang")
    ):
        source = state["v_ang"]

        common = [
            x for x in source.columns
            if x in n.buses_t.v_ang.columns
        ]

        if common:
            n.buses_t.v_ang.loc[
                SNAPSHOT,
                common
            ] = source.loc[
                SNAPSHOT,
                common
            ]

    if (
        "v_ang_pu" in state
        and hasattr(n.buses_t, "v_ang_pu")
    ):
        source = state["v_ang_pu"]

        common = [
            x for x in source.columns
            if x in n.buses_t.v_ang_pu.columns
        ]

        if common:
            n.buses_t.v_ang_pu.loc[
                SNAPSHOT,
                common
            ] = source.loc[
                SNAPSHOT,
                common
            ]


def reset_voltage_state(n):
    """
    Reset AC voltage state to the network's normal initial state.

    This is used only when no previous converged state is available.
    """

    # We intentionally do not manufacture voltages.
    # PyPSA will construct its normal initial state.
    pass


# ============================================================================
# NUMERICAL VALIDATION
# ============================================================================

def voltage_state_is_valid(n):
    """
    Detect NaN/Inf/numerical explosion after PF.
    """

    try:

        vm = n.buses_t.v_mag_pu.loc[
            SNAPSHOT
        ].astype(float)

    except Exception:
        return False, "Voltage magnitude table unavailable."

    values = vm.to_numpy(dtype=float)

    if not np.all(np.isfinite(values)):
        return False, "NaN/Inf voltage magnitude."

    if np.any(np.abs(values) > MAX_VALID_V):
        return False, "Voltage magnitude outside numerical range."

    if np.any(np.abs(values) < MIN_VALID_V):
        return False, "Voltage magnitude collapsed."

    # Check angles if available.
    if hasattr(n.buses_t, "v_ang"):

        va = n.buses_t.v_ang.loc[
            SNAPSHOT
        ].astype(float).to_numpy()

        if not np.all(np.isfinite(va)):
            return False, "NaN/Inf voltage angle."

        if np.any(np.abs(va) > MAX_ABS_VALUE):
            return False, "Voltage angle numerically exploded."

    return True, ""


def calculate_powerflow_error(n):
    """
    Calculate a simple post-PF active-power residual.

    PyPSA may expose different result structures depending on version,
    so this is intentionally conservative.
    """

    try:

        gen = float(
            n.generators_t.p.loc[
                SNAPSHOT
            ].sum()
        )

        load = float(
            n.loads_t.p.loc[
                SNAPSHOT
            ].sum()
        )

        return abs(gen - load)

    except Exception:
        return float("nan")


# ============================================================================
# POWER FLOW
# ============================================================================

def run_pf(n, voltage_state=None):
    """
    Run nonlinear AC PF.

    If voltage_state exists, restore it BEFORE solving and request
    PyPSA to use the existing state as a seed.

    IMPORTANT:
    If the PF fails, the caller MUST NOT retain the resulting voltage
    state as a continuation seed.
    """

    if voltage_state is not None:
        restore_voltage_state(
            n,
            voltage_state,
        )

    else:
        reset_voltage_state(n)

    converged = False

    error_message = ""

    try:

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            result = n.pf(
                snapshots=[SNAPSHOT],
                use_seed=(
                    voltage_state is not None
                ),
                x_tol=X_TOL,
            )

        # --------------------------------------------------------------
        # PyPSA returns a DataFrame/tuple-like object containing
        # convergence information depending on version.
        # --------------------------------------------------------------

        try:

            conv = result["converged"]

            if isinstance(conv, pd.DataFrame):
                converged = bool(
                    conv.loc[
                        SNAPSHOT
                    ].iloc[0]
                    if conv.shape[1] > 0
                    else conv.loc[SNAPSHOT]
                )

            elif isinstance(conv, pd.Series):
                converged = bool(
                    conv.loc[SNAPSHOT]
                )

            else:
                converged = bool(
                    np.asarray(conv).reshape(-1)[0]
                )

        except Exception:

            # If the result structure differs, use voltage validity
            # as a secondary criterion.
            converged = True

    except Exception as exc:

        error_message = str(exc)
        converged = False

    # --------------------------------------------------------------
    # Numerical validation.
    # --------------------------------------------------------------

    if converged:

        valid, reason = voltage_state_is_valid(n)

        if not valid:

            converged = False
            error_message = reason

    return converged, error_message


# ============================================================================
# RESULT METRICS
# ============================================================================

def get_line_loading(n):
    """
    Return line loading percentage and maximum line.
    """

    if n.lines.empty:
        return float("nan"), None

    try:

        loading = (
            n.lines_t.p0
            .loc[SNAPSHOT]
            .abs()
            / n.lines.s_nom
            * 100.0
        )

        loading = loading.replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        if loading.empty:
            return float("nan"), None

        idx = loading.idxmax()

        return float(loading.loc[idx]), idx

    except Exception:
        return float("nan"), None


def get_transformer_loading(n):
    """
    Return maximum transformer loading.
    """

    if n.transformers.empty:
        return float("nan"), None

    try:

        loading0 = (
            n.transformers_t.p0
            .loc[SNAPSHOT]
            .abs()
            / n.transformers.s_nom
            * 100.0
        )

        loading1 = (
            n.transformers_t.p1
            .loc[SNAPSHOT]
            .abs()
            / n.transformers.s_nom
            * 100.0
        )

        loading = pd.concat(
            [loading0, loading1],
            axis=1,
        ).max(axis=1)

        loading = loading.replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        if loading.empty:
            return float("nan"), None

        idx = loading.idxmax()

        return float(loading.loc[idx]), idx

    except Exception:
        return float("nan"), None


def get_voltage_metrics(n):
    vm = (
        n.buses_t.v_mag_pu
        .loc[SNAPSHOT]
        .astype(float)
    )

    min_bus = vm.idxmin()
    max_bus = vm.idxmax()

    return (
        float(vm.min()),
        min_bus,
        float(vm.max()),
        max_bus,
    )


def get_overloaded_lines(n, threshold=100.0):
    """
    Number of lines above 100%.
    """

    if n.lines.empty:
        return 0

    try:

        loading = (
            n.lines_t.p0
            .loc[SNAPSHOT]
            .abs()
            / n.lines.s_nom
            * 100.0
        )

        return int(
            (loading > threshold).sum()
        )

    except Exception:
        return 0


# ============================================================================
# REINFORCEMENT
# ============================================================================

def reinforce_line(n, candidate, multiplier):
    """
    Apply reinforcement to exactly one candidate line.
    """

    if candidate not in n.lines.index:
        raise KeyError(
            f"Candidate line not found: {candidate}"
        )

    original = float(
        n.lines.at[
            candidate,
            "s_nom"
        ]
    )

    new_value = (
        original
        * multiplier
    )

    n.lines.at[
        candidate,
        "s_nom"
    ] = new_value

    return original, new_value


# ============================================================================
# SINGLE PF REPORT
# ============================================================================

def print_pf_result(
    n,
    candidate,
    multiplier,
    lam,
    reinforced_original,
    reinforced_new,
):
    """
    Print a compact S3.5 result.
    """

    min_v, min_bus, max_v, max_bus = (
        get_voltage_metrics(n)
    )

    max_loading, max_line = (
        get_line_loading(n)
    )

    transformer_loading, transformer = (
        get_transformer_loading(n)
    )

    overloaded = get_overloaded_lines(n)

    reinforced_loading = float("nan")

    try:

        reinforced_loading = (
            abs(
                n.lines_t.p0
                .loc[
                    SNAPSHOT,
                    candidate
                ]
            )
            / n.lines.at[
                candidate,
                "s_nom"
            ]
            * 100.0
        )

    except Exception:
        pass

    print()
    print("RESULT")
    print("-" * 110)

    print(
        f"Converged       : TRUE"
    )

    print(
        f"Lambda          : {lam:.6f}"
    )

    print(
        f"Min V magnitude : {min_v:.6f} pu"
    )

    print(
        f"Min-V bus       : {min_bus}"
    )

    print(
        f"Max V magnitude : {max_v:.6f} pu"
    )

    print(
        f"Max-V bus       : {max_bus}"
    )

    print(
        f"Max line loading: {max_loading:.6f} %"
    )

    print(
        f"Max loaded line : {max_line}"
    )

    print(
        f"Reinforced line : {reinforced_loading:.6f} %"
    )

    print(
        f"Overloaded lines: {overloaded}"
    )

    print(
        f"Max transformer : {transformer_loading:.6f} %"
    )

    print(
        f"Transformer     : {transformer}"
    )


# ============================================================================
# FAILED PF REPORT
# ============================================================================

def print_failed_pf(lam, reason):
    print()
    print("RESULT")
    print("-" * 110)
    print("Converged       : FALSE")
    print(f"Lambda          : {lam:.6f}")
    print("Voltage state   : INVALID / DIVERGED")

    if reason:
        print(f"Reason          : {reason}")


# ============================================================================
# ADAPTIVE CONTINUATION
# ============================================================================

def adaptive_continuation(
    n,
    reference,
    candidate,
    multiplier,
    reinforced_original,
    reinforced_new,
):
    """
    Controlled continuation from lambda=0.9 to lambda=1.0.

    Core rule:

        ONLY a CONVERGED solution may become the next seed.

    If a proposed lambda fails:

        1. discard the failed voltage state
        2. restore last converged state
        3. reduce lambda step
        4. retry

    This prevents the exact failure seen in V5 where the enormous
    divergent state at lambda=0.975 becomes part of the continuation.
    """

    START = INITIAL_LAMBDAS[0]
    TARGET = INITIAL_LAMBDAS[-1]

    current_lambda = START

    # --------------------------------------------------------------
    # Start from a clean network state.
    # --------------------------------------------------------------

    start_generation, start_load, start_mismatch, _, _ = (
        apply_lambda(
            n,
            reference,
            current_lambda,
        )
    )

    print()
    print(
        f"Generation : {start_generation:.6f} MW"
    )

    print(
        f"Load       : {start_load:.6f} MW"
    )

    print(
        f"Mismatch   : {start_mismatch:.12f} MW"
    )

    print(
        f"Scaled fixed generation : "
        f"{current_lambda * reference['fixed_generation']:.6f} MW"
    )

    print(
        f"Required non-wind       : "
        f"{current_lambda * reference['required_non_wind']:.6f} MW"
    )

    print(
        "Initialization: NETWORK INITIAL STATE"
    )

    # --------------------------------------------------------------
    # First solve.
    # --------------------------------------------------------------

    converged, reason = run_pf(
        n,
        voltage_state=None,
    )

    if not converged:

        print_failed_pf(
            current_lambda,
            reason,
        )

        return {
            "success": False,
            "final_lambda": None,
            "results": [],
        }

    print_pf_result(
        n,
        candidate,
        multiplier,
        current_lambda,
        reinforced_original,
        reinforced_new,
    )

    # --------------------------------------------------------------
    # Store last VALID state.
    # --------------------------------------------------------------

    last_valid_state = capture_voltage_state(n)

    last_valid_lambda = current_lambda

    results = [
        {
            "lambda": current_lambda,
            "converged": True,
        }
    ]

    # --------------------------------------------------------------
    # Adaptive continuation.
    # --------------------------------------------------------------

    step = MAX_STEP

    while current_lambda < TARGET - 1e-12:

        proposed = min(
            current_lambda + step,
            TARGET,
        )

        attempt = 0
        success = False

        while attempt <= MAX_RETRIES:

            attempt += 1

            print()
            print(
                "-" * 110
            )

            print(
                f"RUNNING AC POWER FLOW — "
                f"lambda = {proposed:.6f}"
            )

            generation, load, mismatch, fixed, required = (
                apply_lambda(
                    n,
                    reference,
                    proposed,
                )
            )

            print(
                f"Generation : {generation:.6f} MW"
            )

            print(
                f"Load       : {load:.6f} MW"
            )

            print(
                f"Mismatch   : {mismatch:.12f} MW"
            )

            print(
                f"Scaled fixed generation : "
                f"{fixed:.6f} MW"
            )

            print(
                f"Required non-wind       : "
                f"{required:.6f} MW"
            )

            print(
                "Initialization: "
                "LAST CONVERGED SOLUTION"
            )

            # ------------------------------------------------------
            # IMPORTANT:
            #
            # Always restore LAST VALID state.
            #
            # Never use the failed state from the previous attempt.
            # ------------------------------------------------------

            restore_voltage_state(
                n,
                last_valid_state,
            )

            converged, reason = run_pf(
                n,
                voltage_state=last_valid_state,
            )

            if converged:

                success = True

                print_pf_result(
                    n,
                    candidate,
                    multiplier,
                    proposed,
                    reinforced_original,
                    reinforced_new,
                )

                # --------------------------------------------------
                # Commit this state.
                # --------------------------------------------------

                current_lambda = proposed

                last_valid_lambda = current_lambda

                last_valid_state = (
                    capture_voltage_state(n)
                )

                results.append(
                    {
                        "lambda": current_lambda,
                        "converged": True,
                    }
                )

                # --------------------------------------------------
                # Successful step:
                # gently increase step.
                # --------------------------------------------------

                step = min(
                    step * STEP_GROWTH,
                    MAX_STEP,
                )

                break

            # ------------------------------------------------------
            # FAILED STEP
            # ------------------------------------------------------

            print_failed_pf(
                proposed,
                reason,
            )

            # ------------------------------------------------------
            # CRITICAL:
            #
            # Immediately throw away the failed voltage state.
            # Restore the last known good state.
            # ------------------------------------------------------

            restore_voltage_state(
                n,
                last_valid_state,
            )

            new_step = (
                step
                * STEP_SHRINK
            )

            if new_step < MIN_STEP:

                print()
                print(
                    "Continuation cannot proceed:"
                )

                print(
                    f"Last converged lambda : "
                    f"{last_valid_lambda:.6f}"
                )

                print(
                    f"Failed target lambda   : "
                    f"{proposed:.6f}"
                )

                print(
                    f"Minimum step reached   : "
                    f"{new_step:.6f}"
                )

                return {
                    "success": False,
                    "final_lambda": last_valid_lambda,
                    "results": results,
                }

            step = new_step

            proposed = min(
                current_lambda + step,
                TARGET,
            )

            print()
            print(
                f"Continuation RETRY"
            )

            print(
                f"Last valid lambda : "
                f"{last_valid_lambda:.6f}"
            )

            print(
                f"Reduced step      : "
                f"{step:.6f}"
            )

            print(
                f"Retry lambda      : "
                f"{proposed:.6f}"
            )

        if not success:

            return {
                "success": False,
                "final_lambda": last_valid_lambda,
                "results": results,
            }

    return {
        "success": True,
        "final_lambda": last_valid_lambda,
        "results": results,
    }


# ============================================================================
# SUMMARY STORAGE
# ============================================================================

def summarize_case(
    n,
    candidate,
    multiplier,
    continuation_result,
):
    final_lambda = (
        continuation_result["final_lambda"]
    )

    success = (
        continuation_result["success"]
        and final_lambda is not None
        and abs(final_lambda - 1.0) < 1e-8
    )

    row = {
        "candidate": candidate,
        "multiplier": multiplier,
        "converged_to_lambda_1": success,
        "final_lambda": final_lambda,
        "reinforced_original_s_nom": float(
            n.lines.at[
                candidate,
                "s_nom"
            ]
        ) / multiplier,
        "reinforced_final_s_nom": float(
            n.lines.at[
                candidate,
                "s_nom"
            ]
        ),
    }

    if success:

        try:

            min_v, min_bus, max_v, max_bus = (
                get_voltage_metrics(n)
            )

            max_loading, max_line = (
                get_line_loading(n)
            )

            transformer_loading, transformer = (
                get_transformer_loading(n)
            )

            overloaded = get_overloaded_lines(n)

            reinforced_loading = (
                abs(
                    n.lines_t.p0
                    .loc[
                        SNAPSHOT,
                        candidate
                    ]
                )
                / n.lines.at[
                    candidate,
                    "s_nom"
                ]
                * 100.0
            )

            row.update(
                {
                    "min_v_pu": min_v,
                    "min_v_bus": min_bus,
                    "max_v_pu": max_v,
                    "max_v_bus": max_bus,
                    "max_line_loading_pct": max_loading,
                    "max_loaded_line": max_line,
                    "reinforced_line_loading_pct":
                        reinforced_loading,
                    "overloaded_lines":
                        overloaded,
                    "max_transformer_loading_pct":
                        transformer_loading,
                    "max_transformer":
                        transformer,
                }
            )

        except Exception as exc:

            row["metric_error"] = str(exc)

    return row


# ============================================================================
# MAIN
# ============================================================================

def main():

    header(
        "S3.5 INDIVIDUAL-LINE REINFORCEMENT — "
        "CONTROLLED AC CONTINUATION — BALANCE CORRECTED V5"
    )

    print()
    print(
        f"Network : {NETWORK_PATH}"
    )

    print(
        f"Snapshot: {SNAPSHOT}"
    )

    print()
    print(
        "Continuation scales:"
    )

    print(
        INITIAL_LAMBDAS
    )

    print()
    print(
        "Candidates:"
    )

    for candidate in CANDIDATES:
        print(
            f"  {candidate}"
        )

    print()
    print(
        "Reinforcements:"
    )

    for multiplier in REINFORCEMENTS:
        print(
            f"  {multiplier:.2f}x"
        )

    # ------------------------------------------------------------------
    # Load ORIGINAL network.
    # ------------------------------------------------------------------

    original = load_network()

    # ------------------------------------------------------------------
    # Original balance diagnostics.
    # ------------------------------------------------------------------

    original_generation, original_load, original_mismatch = (
        calculate_original_balance(
            original
        )
    )

    header(
        "ORIGINAL POWER BALANCE"
    )

    print(
        f"Generation : {original_generation:.6f} MW"
    )

    print(
        f"Load       : {original_load:.6f} MW"
    )

    print(
        f"Mismatch   : {original_mismatch:.6f} MW"
    )

    # ------------------------------------------------------------------
    # Build BALANCED S2 reference.
    # ------------------------------------------------------------------

    reference_network = original.copy()

    reference = build_balanced_reference(
        reference_network
    )

    capture_reference_values(
        reference_network,
        reference,
    )

    balanced_generation = float(
        reference_network
        .generators_t
        .p_set
        .loc[SNAPSHOT]
        .sum()
    )

    balanced_load = float(
        reference_network
        .loads_t
        .p_set
        .loc[SNAPSHOT]
        .sum()
    )

    balanced_mismatch = (
        balanced_generation
        - balanced_load
    )

    header(
        "BALANCED S2 REFERENCE"
    )

    print(
        f"Original total generation   : "
        f"{original_generation:.6f} MW"
    )

    print(
        f"Load                        : "
        f"{reference['load']:.6f} MW"
    )

    print(
        f"Original non-wind generation: "
        f"{reference['original_non_wind']:.6f} MW"
    )

    print(
        f"Fixed generation            : "
        f"{reference['fixed_generation']:.6f} MW"
    )

    print(
        f"Required non-wind generation: "
        f"{reference['required_non_wind']:.6f} MW"
    )

    print(
        f"Balanced generation         : "
        f"{balanced_generation:.6f} MW"
    )

    print(
        f"Load                        : "
        f"{balanced_load:.6f} MW"
    )

    print(
        f"Reference mismatch          : "
        f"{balanced_mismatch:.12f} MW"
    )

    if abs(balanced_mismatch) > BALANCE_TOL:

        raise RuntimeError(
            "Balanced S2 reference is not balanced."
        )

    # ------------------------------------------------------------------
    # Reference dispatch report.
    # ------------------------------------------------------------------

    header(
        "REFERENCE GENERATOR DISPATCH"
    )

    for g in reference["wind"]:

        print(
            f"{g:<55}"
            f"{'wind':<25}"
            f"{get_generator_output(reference_network, g):>12.6f} MW"
        )

    for g in reference["non_wind"]:

        print(
            f"{g:<55}"
            f"{'non_wind':<25}"
            f"{get_generator_output(reference_network, g):>12.6f} MW"
        )

    for g in reference["interconnectors"]:

        print(
            f"{g:<55}"
            f"{'interconnector':<25}"
            f"{get_generator_output(reference_network, g):>12.6f} MW"
        )

    # ------------------------------------------------------------------
    # Results.
    # ------------------------------------------------------------------

    all_results = []

    # ------------------------------------------------------------------
    # Each candidate starts from the SAME balanced S2 reference.
    # ------------------------------------------------------------------

    for candidate in CANDIDATES:

        if candidate not in original.lines.index:

            print()
            print(
                f"WARNING: Candidate not found: {candidate}"
            )

            continue

        for multiplier in REINFORCEMENTS:

            header(
                f"TEST — {candidate} — "
                f"{multiplier:.2f}x REINFORCEMENT"
            )

            # ----------------------------------------------------------
            # IMPORTANT:
            #
            # Start every individual reinforcement test from the
            # identical balanced reference network.
            #
            # No previous candidate's reinforcement can leak into
            # another candidate.
            # ----------------------------------------------------------

            n = reference_network.copy()

            original_s_nom, new_s_nom = reinforce_line(
                n,
                candidate,
                multiplier,
            )

            subheader(
                "REINFORCEMENT"
            )

            print(
                f"Original s_nom : "
                f"{original_s_nom:.6f} MW"
            )

            print(
                f"Multiplier     : "
                f"{multiplier:.2f}x"
            )

            print(
                f"New s_nom      : "
                f"{new_s_nom:.6f} MW"
            )

            # ----------------------------------------------------------
            # Run adaptive AC continuation.
            # ----------------------------------------------------------

            result = adaptive_continuation(
                n=n,
                reference=reference,
                candidate=candidate,
                multiplier=multiplier,
                reinforced_original=original_s_nom,
                reinforced_new=new_s_nom,
            )

            # ----------------------------------------------------------
            # Store summary.
            # ----------------------------------------------------------

            summary = summarize_case(
                n=n,
                candidate=candidate,
                multiplier=multiplier,
                continuation_result=result,
            )

            all_results.append(
                summary
            )

            # ----------------------------------------------------------
            # Case completion.
            # ----------------------------------------------------------

            print()
            print(
                "=" * WIDTH
            )

            if result["success"]:

                print(
                    f"CASE COMPLETE: {candidate} "
                    f"{multiplier:.2f}x "
                    f"reached lambda=1.000"
                )

            else:

                print(
                    f"CASE INCOMPLETE: {candidate} "
                    f"{multiplier:.2f}x "
                    f"stopped at lambda="
                    f"{result['final_lambda']}"
                )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    header(
        "S3.5 FINAL SUMMARY"
    )

    if not all_results:

        print(
            "No candidate results were produced."
        )

        return

    summary_df = pd.DataFrame(
        all_results
    )

    # --------------------------------------------------------------
    # Print compact summary.
    # --------------------------------------------------------------

    display_columns = [
        "candidate",
        "multiplier",
        "converged_to_lambda_1",
        "final_lambda",
        "min_v_pu",
        "max_v_pu",
        "max_line_loading_pct",
        "reinforced_line_loading_pct",
        "overloaded_lines",
        "max_transformer_loading_pct",
    ]

    existing_columns = [
        c for c in display_columns
        if c in summary_df.columns
    ]

    print(
        summary_df[
            existing_columns
        ].to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # Save CSV.
    # ------------------------------------------------------------------

    output_path = Path(
        "data/processed/"
        "s3_5_individual_line_continuation_results_v5.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        f"Results saved to:"
    )

    print(
        output_path
    )

    header(
        "S3.5 COMPLETE"
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()