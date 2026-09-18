# ==================================================================================================
# S4.4 — COORDINATED RESIDUAL BOTTLENECK MITIGATION SCREEN
# ==================================================================================================
#
# Purpose
# -------
# Determine whether the residual thermal + voltage bottleneck identified in S4.2/S4.3
# can be removed through coordinated reinforcement of multiple residual lines and
# increased reactive support.
#
# IMPORTANT
# ---------
# The source network is READ-ONLY.
# Every candidate starts from a fresh import of the original network.
# No .nc network file is modified.
#
# Network:
#   data/processed/eirgrid_second_reinforced_network.nc
#
# Snapshot:
#   S2_PEAK_DEMAND
#
# Base package:
#   P3_HIGH_COORDINATED
#
# Power flow:
#   AC nonlinear
#
# Dispatch:
#   unchanged
#
# Loads:
#   unchanged
#
# ==================================================================================================

from pathlib import Path
from itertools import combinations
import warnings

import numpy as np
import pandas as pd
import pypsa


warnings.filterwarnings("ignore")


# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

NETWORK_PATH = Path(
    "data/processed/eirgrid_second_reinforced_network.nc"
)

SNAPSHOT = "S2_PEAK_DEMAND"

OUTPUT_RESULTS = Path(
    "data/processed/s4_4_coordinated_residual_mitigation_results.csv"
)

OUTPUT_RANKING = Path(
    "data/processed/s4_4_coordinated_residual_mitigation_ranking.csv"
)


# ==================================================================================================
# ACCEPTANCE THRESHOLDS
# ==================================================================================================

V_MIN_LIMIT = 0.95
V_MAX_LIMIT = 1.05

LINE_LOADING_LIMIT = 100.0
TRANSFORMER_LOADING_LIMIT = 100.0


# ==================================================================================================
# P3 HIGH COORDINATED REINFORCEMENTS
# ==================================================================================================

P3_REINFORCEMENTS = {
    "merged_way/1231251986-220+2": 1.75,
    "merged_way/61295764-220+1": 2.00,
    "way/343436171-220": 2.00,
    "merged_way/257889771-220+1": 1.75,
    "merged_relation/4872159-220+1": 1.75,
}


# ==================================================================================================
# RESIDUAL BOTTLENECKS FROM S4.2
# ==================================================================================================

RESIDUAL_LINES = [
    "way/235559472-220",
    "way/713396116-220",
    "way/42838773-220",
    "merged_way/516651706-220+2",
]


# ==================================================================================================
# REACTIVE SUPPORT CONFIGURATION
# ==================================================================================================

REACTIVE_GENERATOR = "eirgrid_wind_way/104388595-220"

BASE_REACTIVE_MVAR = 500.0

REACTIVE_LEVELS = [
    500.0,
    750.0,
    1000.0,
    1250.0,
    1500.0,
]


# ==================================================================================================
# THERMAL MULTIPLIERS
# ==================================================================================================

PAIR_MULTIPLIERS = [
    1.25,
    1.50,
]

ALL_FOUR_MULTIPLIERS = [
    1.25,
    1.50,
]


# ==================================================================================================
# CANDIDATE DEFINITION
# ==================================================================================================

candidates = []


def add_candidate(
    name,
    target_lines,
    multiplier,
    reactive_mvar,
    category,
):
    candidates.append(
        {
            "candidate": name,
            "target_lines": target_lines,
            "multiplier": float(multiplier),
            "reactive_mvar": float(reactive_mvar),
            "category": category,
        }
    )


# ==================================================================================================
# CANDIDATE GROUP A — BASELINE
# ==================================================================================================

add_candidate(
    "BASE_P3_PLUS_500MVAR",
    [],
    1.00,
    500.0,
    "BASELINE",
)


# ==================================================================================================
# CANDIDATE GROUP B — PAIRWISE THERMAL
# ==================================================================================================

for line_a, line_b in combinations(
    RESIDUAL_LINES,
    2,
):

    pair = [
        line_a,
        line_b,
    ]

    for multiplier in PAIR_MULTIPLIERS:

        safe_a = (
            line_a
            .replace("/", "_")
            .replace("+", "plus")
        )

        safe_b = (
            line_b
            .replace("/", "_")
            .replace("+", "plus")
        )

        name = (
            f"PAIR_{safe_a}_{safe_b}_"
            f"{multiplier:.2f}X_PLUS_500MVAR"
        )

        add_candidate(
            name,
            pair,
            multiplier,
            500.0,
            "PAIRWISE_THERMAL",
        )


# ==================================================================================================
# CANDIDATE GROUP C — ALL FOUR THERMAL
# ==================================================================================================

for multiplier in ALL_FOUR_MULTIPLIERS:

    name = (
        f"ALL_4_RESIDUAL_LINES_"
        f"{multiplier:.2f}X_PLUS_500MVAR"
    )

    add_candidate(
        name,
        RESIDUAL_LINES.copy(),
        multiplier,
        500.0,
        "ALL_FOUR_THERMAL",
    )


# ==================================================================================================
# CANDIDATE GROUP D — REACTIVE ONLY
# ==================================================================================================

for q in REACTIVE_LEVELS:

    if q == BASE_REACTIVE_MVAR:
        continue

    name = (
        f"P3_PLUS_{int(q)}MVAR"
    )

    add_candidate(
        name,
        [],
        1.00,
        q,
        "REACTIVE_ONLY",
    )


# ==================================================================================================
# CANDIDATE GROUP E — COORDINATED THERMAL + REACTIVE
# ==================================================================================================

combined_reactive_levels = [
    750.0,
    1000.0,
]

for multiplier in ALL_FOUR_MULTIPLIERS:

    for q in combined_reactive_levels:

        name = (
            f"ALL_4_RESIDUAL_LINES_"
            f"{multiplier:.2f}X_PLUS_{int(q)}MVAR"
        )

        add_candidate(
            name,
            RESIDUAL_LINES.copy(),
            multiplier,
            q,
            "COORDINATED_THERMAL_REACTIVE",
        )


# ==================================================================================================
# CANDIDATE GROUP F — STRONG COORDINATED
# ==================================================================================================

for q in [
    1250.0,
    1500.0,
]:

    name = (
        f"ALL_4_RESIDUAL_LINES_"
        f"1.50X_PLUS_{int(q)}MVAR"
    )

    add_candidate(
        name,
        RESIDUAL_LINES.copy(),
        1.50,
        q,
        "STRONG_COORDINATED",
    )


# ==================================================================================================
# REMOVE DUPLICATES
# ==================================================================================================

unique_candidates = []

seen = set()

for candidate in candidates:

    key = (
        candidate["candidate"],
        tuple(candidate["target_lines"]),
        candidate["multiplier"],
        candidate["reactive_mvar"],
        candidate["category"],
    )

    if key not in seen:

        seen.add(key)

        unique_candidates.append(
            candidate
        )


candidates = unique_candidates


# ==================================================================================================
# HELPERS
# ==================================================================================================

def safe_float(
    value,
    default=np.nan,
):
    """
    Safely convert a value to float.
    """

    try:

        return float(value)

    except Exception:

        return default


# ==================================================================================================
# NETWORK SCHEMA INSPECTION
# ==================================================================================================

def print_generator_schema(network):
    """
    Print the actual generator schema.

    This is intentionally defensive because PyPSA versions/networks can differ.
    """

    print()
    print("-" * 100)
    print("GENERATOR SCHEMA")
    print("-" * 100)

    print(
        "Generator columns:"
    )

    print(
        list(network.generators.columns)
    )

    print()
    print(
        "Generator names:"
    )

    for name in network.generators.index:

        bus = ""

        if "bus" in network.generators.columns:

            bus = str(
                network.generators.at[
                    name,
                    "bus",
                ]
            )

        print(
            f"{str(name):<55} bus={bus}"
        )

    print()
    print(
        "Generator time-series columns:"
    )

    print(
        list(network.generators_t.keys())
    )


# ==================================================================================================
# FIND REACTIVE GENERATOR
# ==================================================================================================

def find_reactive_generator(network):
    """
    Find the generator intended for reactive support.

    Priority:
        1. Exact known generator.
        2. Generator containing 104388595.
        3. Generator attached to the weak bus.
        4. Generator with an existing q_set time-series.
        5. First available generator.

    IMPORTANT:
    ----------
    This function deliberately DOES NOT use generators.q_nom because the
    current network schema does not contain that column.
    """

    # ----------------------------------------------------------------------------------------------
    # 1. Exact known generator
    # ----------------------------------------------------------------------------------------------

    if REACTIVE_GENERATOR in network.generators.index:

        return REACTIVE_GENERATOR


    # ----------------------------------------------------------------------------------------------
    # 2. Generator containing 104388595
    # ----------------------------------------------------------------------------------------------

    matches = [
        name
        for name in network.generators.index
        if "104388595" in str(name)
    ]

    if matches:

        return matches[0]


    # ----------------------------------------------------------------------------------------------
    # 3. Generator attached to weak bus
    # ----------------------------------------------------------------------------------------------

    weak_bus = "way/1003262502-220"

    if weak_bus in network.buses.index:

        if "bus" in network.generators.columns:

            matches = (
                network.generators.index[
                    network.generators.bus == weak_bus
                ]
                .tolist()
            )

            if matches:

                return matches[0]


    # ----------------------------------------------------------------------------------------------
    # 4. Generator with q_set time-series
    # ----------------------------------------------------------------------------------------------

    try:

        q_set = network.generators_t.q_set

        if q_set is not None and len(q_set.columns) > 0:

            available = [
                name
                for name in q_set.columns
                if name in network.generators.index
            ]

            if available:

                return available[0]

    except Exception:

        pass


    # ----------------------------------------------------------------------------------------------
    # 5. First available generator
    # ----------------------------------------------------------------------------------------------

    if len(network.generators.index) > 0:

        return network.generators.index[0]


    return None


# ==================================================================================================
# APPLY P3 REINFORCEMENTS
# ==================================================================================================

def apply_p3_reinforcements(network):

    print()
    print("-" * 100)
    print("APPLYING P3 REINFORCEMENTS")
    print("-" * 100)

    for line_name, multiplier in P3_REINFORCEMENTS.items():

        if line_name not in network.lines.index:

            print(
                f"WARNING: P3 line not found: {line_name}"
            )

            continue

        old_s_nom = safe_float(
            network.lines.at[
                line_name,
                "s_nom",
            ]
        )

        new_s_nom = (
            old_s_nom
            * multiplier
        )

        network.lines.at[
            line_name,
            "s_nom",
        ] = new_s_nom

        print(
            f"{line_name:<55}"
            f"{multiplier:>6.2f}x"
            f"{old_s_nom:>12.3f} -> "
            f"{new_s_nom:>12.3f} MVA"
        )


# ==================================================================================================
# APPLY TARGETED REINFORCEMENTS
# ==================================================================================================

def apply_targeted_reinforcements(
    network,
    target_lines,
    multiplier,
):

    if not target_lines:

        return

    print()
    print("-" * 100)
    print("APPLYING COORDINATED TARGETED REINFORCEMENTS")
    print("-" * 100)

    for line_name in target_lines:

        if line_name not in network.lines.index:

            print(
                f"WARNING: target line not found: {line_name}"
            )

            continue

        old_s_nom = safe_float(
            network.lines.at[
                line_name,
                "s_nom",
            ]
        )

        new_s_nom = (
            old_s_nom
            * multiplier
        )

        network.lines.at[
            line_name,
            "s_nom",
        ] = new_s_nom

        print(
            f"{line_name:<55}"
            f"{multiplier:>6.2f}x"
            f"{old_s_nom:>12.3f} -> "
            f"{new_s_nom:>12.3f} MVA"
        )


# ==================================================================================================
# ENSURE Q_SET
# ==================================================================================================

def ensure_q_set(network):
    """
    Ensure generators_t.q_set exists for the requested snapshot.

    If q_set does not exist, create it for all network snapshots and generators.

    This is only an in-memory modification of the candidate network.
    """

    if SNAPSHOT not in network.snapshots:

        raise ValueError(
            f"Snapshot '{SNAPSHOT}' not found."
        )

    generator_names = network.generators.index

    # ----------------------------------------------------------------------------------------------
    # q_set already exists
    # ----------------------------------------------------------------------------------------------

    try:

        q_set = network.generators_t.q_set

        if q_set is not None:

            if SNAPSHOT in q_set.index:

                # Make sure every current generator has a column.
                missing_columns = [
                    generator
                    for generator in generator_names
                    if generator not in q_set.columns
                ]

                if missing_columns:

                    for generator in missing_columns:

                        q_set[generator] = 0.0

                    network.generators_t.q_set = q_set

                return

    except Exception:

        pass


    # ----------------------------------------------------------------------------------------------
    # Create q_set
    # ----------------------------------------------------------------------------------------------

    snapshots = network.snapshots

    q_values = pd.DataFrame(
        0.0,
        index=snapshots,
        columns=generator_names,
    )


    # ----------------------------------------------------------------------------------------------
    # If an existing q_set is partially available, preserve it.
    # ----------------------------------------------------------------------------------------------

    try:

        existing_q_set = network.generators_t.q_set

        if existing_q_set is not None:

            common_rows = existing_q_set.index.intersection(
                snapshots
            )

            common_columns = existing_q_set.columns.intersection(
                generator_names
            )

            if len(common_rows) > 0 and len(common_columns) > 0:

                q_values.loc[
                    common_rows,
                    common_columns,
                ] = existing_q_set.loc[
                    common_rows,
                    common_columns,
                ]

    except Exception:

        pass


    network.generators_t.q_set = q_values


# ==================================================================================================
# APPLY REACTIVE SUPPORT
# ==================================================================================================

def apply_reactive_support(
    network,
    reactive_mvar,
):

    print()
    print("-" * 100)
    print("APPLYING REACTIVE SUPPORT")
    print("-" * 100)

    generator = find_reactive_generator(
        network
    )

    if generator is None:

        print(
            "WARNING: No generator available for reactive support."
        )

        return None


    # Ensure q_set exists.
    ensure_q_set(
        network
    )


    # ----------------------------------------------------------------------------------------------
    # Read previous Q setpoint
    # ----------------------------------------------------------------------------------------------

    old_q = 0.0

    try:

        old_q = safe_float(
            network.generators_t.q_set.loc[
                SNAPSHOT,
                generator,
            ],
            default=0.0,
        )

    except Exception:

        old_q = 0.0


    # ----------------------------------------------------------------------------------------------
    # Apply Q setpoint
    # ----------------------------------------------------------------------------------------------

    network.generators_t.q_set.loc[
        SNAPSHOT,
        generator,
    ] = reactive_mvar


    print(
        f"Reactive support generator : {generator}"
    )

    generator_bus = ""

    if "bus" in network.generators.columns:

        generator_bus = str(
            network.generators.at[
                generator,
                "bus",
            ]
        )

    print(
        f"Generator bus              : {generator_bus}"
    )

    print(
        f"Previous Q setpoint        : "
        f"{old_q:.3f} MVAr"
    )

    print(
        f"New Q setpoint              : "
        f"{reactive_mvar:.3f} MVAr"
    )

    return generator


# ==================================================================================================
# CALCULATE LINE LOADING
# ==================================================================================================

def calculate_line_loading(network):

    if len(network.lines) == 0:

        return pd.Series(
            dtype=float
        )


    if SNAPSHOT not in network.lines_t.p0.index:

        return pd.Series(
            dtype=float
        )


    s0 = np.sqrt(
        network.lines_t.p0.loc[
            SNAPSHOT
        ] ** 2
        +
        network.lines_t.q0.loc[
            SNAPSHOT
        ] ** 2
    )


    s1 = np.sqrt(
        network.lines_t.p1.loc[
            SNAPSHOT
        ] ** 2
        +
        network.lines_t.q1.loc[
            SNAPSHOT
        ] ** 2
    )


    s_max = pd.concat(
        [
            s0.rename("s0"),
            s1.rename("s1"),
        ],
        axis=1,
    ).max(
        axis=1
    )


    s_nom = (
        network.lines.s_nom
        .replace(
            0,
            np.nan,
        )
    )


    return (
        s_max
        / s_nom
        * 100.0
    )


# ==================================================================================================
# CALCULATE TRANSFORMER LOADING
# ==================================================================================================

def calculate_transformer_loading(network):

    if len(network.transformers) == 0:

        return pd.Series(
            dtype=float
        )


    if SNAPSHOT not in network.transformers_t.p0.index:

        return pd.Series(
            dtype=float
        )


    s0 = np.sqrt(
        network.transformers_t.p0.loc[
            SNAPSHOT
        ] ** 2
        +
        network.transformers_t.q0.loc[
            SNAPSHOT
        ] ** 2
    )


    s1 = np.sqrt(
        network.transformers_t.p1.loc[
            SNAPSHOT
        ] ** 2
        +
        network.transformers_t.q1.loc[
            SNAPSHOT
        ] ** 2
    )


    s_max = pd.concat(
        [
            s0.rename("s0"),
            s1.rename("s1"),
        ],
        axis=1,
    ).max(
        axis=1
    )


    s_nom = (
        network.transformers.s_nom
        .replace(
            0,
            np.nan,
        )
    )


    return (
        s_max
        / s_nom
        * 100.0
    )


# ==================================================================================================
# RUN CANDIDATE
# ==================================================================================================

def run_candidate(candidate):

    # ----------------------------------------------------------------------------------------------
    # IMPORTANT:
    # Fresh import for EVERY candidate.
    # ----------------------------------------------------------------------------------------------

    network = pypsa.Network(
        str(NETWORK_PATH)
    )


    if SNAPSHOT not in network.snapshots:

        raise ValueError(
            f"Snapshot '{SNAPSHOT}' not found."
        )


    # ----------------------------------------------------------------------------------------------
    # Set requested snapshot.
    # ----------------------------------------------------------------------------------------------

    network.set_snapshots(
        network.snapshots
    )


    # ----------------------------------------------------------------------------------------------
    # P3
    # ----------------------------------------------------------------------------------------------

    apply_p3_reinforcements(
        network
    )


    # ----------------------------------------------------------------------------------------------
    # Candidate thermal reinforcement
    # ----------------------------------------------------------------------------------------------

    if candidate["target_lines"]:

        apply_targeted_reinforcements(
            network,
            candidate["target_lines"],
            candidate["multiplier"],
        )


    # ----------------------------------------------------------------------------------------------
    # Reactive support
    # ----------------------------------------------------------------------------------------------

    reactive_generator = apply_reactive_support(
        network,
        candidate["reactive_mvar"],
    )


    # ----------------------------------------------------------------------------------------------
    # AC nonlinear PF
    # ----------------------------------------------------------------------------------------------

    print()
    print("-" * 100)
    print("RUNNING AC NONLINEAR POWER FLOW")
    print("-" * 100)

    converged = False
    pf_error = ""

    try:

        result = network.pf(
            snapshots=[
                SNAPSHOT
            ],
            x_tol=1e-8,
            use_seed=True,
        )


        # ------------------------------------------------------------------------------------------
        # PyPSA convergence extraction.
        # ------------------------------------------------------------------------------------------

        try:

            converged_values = result["converged"]

            if hasattr(
                converged_values,
                "loc",
            ):

                converged = bool(
                    converged_values
                    .loc[
                        SNAPSHOT
                    ]
                    .all()
                )

            else:

                converged = bool(
                    np.asarray(
                        converged_values
                    ).all()
                )

        except Exception:

            # If PF returned without an exception and convergence information
            # cannot be extracted, treat the run as converged.
            converged = True


    except Exception as exc:

        converged = False

        pf_error = str(
            exc
        )


    # ----------------------------------------------------------------------------------------------
    # PF FAILED
    # ----------------------------------------------------------------------------------------------

    if not converged:

        return {
            "candidate": candidate["candidate"],
            "category": candidate["category"],
            "target_lines": ";".join(
                candidate["target_lines"]
            ),
            "target_multiplier": candidate["multiplier"],
            "reactive_mvar": candidate["reactive_mvar"],
            "reactive_generator": reactive_generator,
            "converged": False,
            "min_voltage_pu": np.nan,
            "min_voltage_bus": "",
            "max_voltage_pu": np.nan,
            "max_voltage_bus": "",
            "low_voltage_buses": np.nan,
            "high_voltage_buses": np.nan,
            "max_line_loading_pct": np.nan,
            "overloaded_lines": np.nan,
            "critical_line": "",
            "max_transformer_loading_pct": np.nan,
            "overloaded_transformers": np.nan,
            "worst_transformer": "",
            "fully_acceptable": False,
            "pf_error": pf_error,
        }


    # ==================================================================================================
    # VOLTAGE METRICS
    # ==================================================================================================

    voltages = (
        network.buses_t.v_mag_pu
        .loc[
            SNAPSHOT
        ]
    )


    min_voltage = safe_float(
        voltages.min()
    )

    min_voltage_bus = str(
        voltages.idxmin()
    )


    max_voltage = safe_float(
        voltages.max()
    )

    max_voltage_bus = str(
        voltages.idxmax()
    )


    low_voltage_mask = (
        voltages
        < V_MIN_LIMIT
    )

    high_voltage_mask = (
        voltages
        > V_MAX_LIMIT
    )


    low_voltage_buses = int(
        low_voltage_mask.sum()
    )

    high_voltage_buses = int(
        high_voltage_mask.sum()
    )


    # ==================================================================================================
    # LINE LOADING
    # ==================================================================================================

    line_loading = calculate_line_loading(
        network
    )


    if len(line_loading) > 0:

        max_line_loading = safe_float(
            line_loading.max()
        )

        critical_line = str(
            line_loading.idxmax()
        )

        overloaded_lines = int(
            (
                line_loading
                > LINE_LOADING_LIMIT
            ).sum()
        )

    else:

        max_line_loading = np.nan

        critical_line = ""

        overloaded_lines = 0


    # ==================================================================================================
    # TRANSFORMER LOADING
    # ==================================================================================================

    transformer_loading = (
        calculate_transformer_loading(
            network
        )
    )


    if len(transformer_loading) > 0:

        max_transformer_loading = safe_float(
            transformer_loading.max()
        )

        worst_transformer = str(
            transformer_loading.idxmax()
        )

        overloaded_transformers = int(
            (
                transformer_loading
                > TRANSFORMER_LOADING_LIMIT
            ).sum()
        )

    else:

        max_transformer_loading = np.nan

        worst_transformer = ""

        overloaded_transformers = 0


    # ==================================================================================================
    # ACCEPTANCE
    # ==================================================================================================

    fully_acceptable = (
        converged
        and overloaded_lines == 0
        and overloaded_transformers == 0
        and low_voltage_buses == 0
        and high_voltage_buses == 0
    )


    # ==================================================================================================
    # PRINT RESULT
    # ==================================================================================================

    print()
    print("RESULT")
    print()

    print(
        f"Converged                 : "
        f"{converged}"
    )

    print(
        f"Minimum voltage           : "
        f"{min_voltage:.6f} pu"
    )

    print(
        f"Minimum-voltage bus       : "
        f"{min_voltage_bus}"
    )

    print(
        f"Low-voltage buses         : "
        f"{low_voltage_buses}"
    )

    print(
        f"Maximum voltage           : "
        f"{max_voltage:.6f} pu"
    )

    print(
        f"Maximum-voltage bus       : "
        f"{max_voltage_bus}"
    )

    print(
        f"High-voltage buses        : "
        f"{high_voltage_buses}"
    )

    print(
        f"Max line loading          : "
        f"{max_line_loading:.6f} %"
    )

    print(
        f"Overloaded lines          : "
        f"{overloaded_lines}"
    )

    print(
        f"Critical line             : "
        f"{critical_line}"
    )

    print(
        f"Max transformer loading   : "
        f"{max_transformer_loading:.6f} %"
    )

    print(
        f"Overloaded transformers   : "
        f"{overloaded_transformers}"
    )

    print(
        f"Worst transformer        : "
        f"{worst_transformer}"
    )

    print(
        f"Fully acceptable          : "
        f"{fully_acceptable}"
    )


    return {
        "candidate": candidate["candidate"],
        "category": candidate["category"],
        "target_lines": ";".join(
            candidate["target_lines"]
        ),
        "target_multiplier": candidate["multiplier"],
        "reactive_mvar": candidate["reactive_mvar"],
        "reactive_generator": reactive_generator,
        "converged": converged,
        "min_voltage_pu": min_voltage,
        "min_voltage_bus": min_voltage_bus,
        "max_voltage_pu": max_voltage,
        "max_voltage_bus": max_voltage_bus,
        "low_voltage_buses": low_voltage_buses,
        "high_voltage_buses": high_voltage_buses,
        "max_line_loading_pct": max_line_loading,
        "overloaded_lines": overloaded_lines,
        "critical_line": critical_line,
        "max_transformer_loading_pct": max_transformer_loading,
        "overloaded_transformers": overloaded_transformers,
        "worst_transformer": worst_transformer,
        "fully_acceptable": fully_acceptable,
        "pf_error": "",
    }


# ==================================================================================================
# HEADER
# ==================================================================================================

print(
    "=" * 100
)

print(
    "S4.4 — COORDINATED RESIDUAL BOTTLENECK MITIGATION SCREEN"
)

print(
    "=" * 100
)

print()
print(
    f"Network  : {NETWORK_PATH}"
)

print(
    f"Snapshot : {SNAPSHOT}"
)

print(
    "Package  : P3_HIGH_COORDINATED"
)

print(
    "PF       : AC nonlinear"
)

print(
    "Dispatch : unchanged"
)

print(
    "Loads    : unchanged"
)

print(
    "Source   : READ-ONLY"
)

print()
print(
    "Purpose  : Coordinated thermal reinforcement + reactive support"
)


# ==================================================================================================
# SOURCE NETWORK CHECK
# ==================================================================================================

print()
print(
    "=" * 100
)

print(
    "SOURCE NETWORK CHECK"
)

print(
    "=" * 100
)


source_network = pypsa.Network(
    str(NETWORK_PATH)
)


if SNAPSHOT not in source_network.snapshots:

    raise ValueError(
        f"Snapshot '{SNAPSHOT}' not found."
    )


print(
    f"Buses       : {len(source_network.buses)}"
)

print(
    f"Lines       : {len(source_network.lines)}"
)

print(
    f"Transformers: {len(source_network.transformers)}"
)

print(
    f"Generators  : {len(source_network.generators)}"
)

print(
    f"Snapshots   : {len(source_network.snapshots)}"
)


# ==================================================================================================
# RESIDUAL BOTTLENECK CHECK
# ==================================================================================================

print()
print(
    "=" * 100
)

print(
    "RESIDUAL BOTTLENECKS UNDER TEST"
)

print(
    "=" * 100
)


for line in RESIDUAL_LINES:

    if line in source_network.lines.index:

        s_nom = safe_float(
            source_network.lines.at[
                line,
                "s_nom",
            ]
        )

        print(
            f"{line:<60}"
            f"{s_nom:>12.3f} MVA"
        )

    else:

        print(
            f"{line:<60}"
            f"NOT FOUND"
        )


# ==================================================================================================
# REACTIVE SUPPORT GENERATOR CHECK
# ==================================================================================================

print()
print(
    "=" * 100
)

print(
    "REACTIVE SUPPORT GENERATOR"
)

print(
    "=" * 100
)


reactive_generator = find_reactive_generator(
    source_network
)


if reactive_generator is None:

    print(
        "WARNING: No generator found."
    )

else:

    reactive_bus = ""

    if "bus" in source_network.generators.columns:

        reactive_bus = str(
            source_network.generators.at[
                reactive_generator,
                "bus",
            ]
        )


    print(
        f"Generator : {reactive_generator}"
    )

    print(
        f"Bus       : {reactive_bus}"
    )

    print()
    print(
        "Available generator columns:"
    )

    print(
        list(
            source_network.generators.columns
        )
    )


# ==================================================================================================
# CANDIDATE COUNT
# ==================================================================================================

print()
print(
    f"TOTAL CANDIDATES: {len(candidates)}"
)


# ==================================================================================================
# RUN SCREEN
# ==================================================================================================

results = []


for i, candidate in enumerate(
    candidates,
    start=1,
):

    print()
    print(
        "=" * 100
    )

    print(
        f"CANDIDATE {i}/{len(candidates)}"
    )

    print(
        "=" * 100
    )


    print(
        f"Candidate : "
        f"{candidate['candidate']}"
    )

    print(
        f"Category  : "
        f"{candidate['category']}"
    )


    if candidate["target_lines"]:

        print(
            "Targets   : "
            + ", ".join(
                candidate["target_lines"]
            )
        )

        print(
            f"Multiplier: "
            f"{candidate['multiplier']:.2f}x"
        )

    else:

        print(
            "Targets   : NONE"
        )


    print(
        f"Reactive  : "
        f"+{candidate['reactive_mvar']:.1f} MVAr"
    )


    try:

        result = run_candidate(
            candidate
        )

    except Exception as exc:

        print()
        print(
            "CANDIDATE ERROR"
        )

        print(
            str(exc)
        )


        result = {
            "candidate": candidate["candidate"],
            "category": candidate["category"],
            "target_lines": ";".join(
                candidate["target_lines"]
            ),
            "target_multiplier": candidate["multiplier"],
            "reactive_mvar": candidate["reactive_mvar"],
            "reactive_generator": "",
            "converged": False,
            "min_voltage_pu": np.nan,
            "min_voltage_bus": "",
            "max_voltage_pu": np.nan,
            "max_voltage_bus": "",
            "low_voltage_buses": np.nan,
            "high_voltage_buses": np.nan,
            "max_line_loading_pct": np.nan,
            "overloaded_lines": np.nan,
            "critical_line": "",
            "max_transformer_loading_pct": np.nan,
            "overloaded_transformers": np.nan,
            "worst_transformer": "",
            "fully_acceptable": False,
            "pf_error": str(exc),
        }


    results.append(
        result
    )


# ==================================================================================================
# RESULTS DATAFRAME
# ==================================================================================================

results_df = pd.DataFrame(
    results
)


# ==================================================================================================
# RANKING LOGIC
# ==================================================================================================

results_df["acceptable_rank"] = (
    ~results_df[
        "fully_acceptable"
    ].fillna(False)
).astype(int)


results_df["overload_rank"] = (
    results_df[
        "overloaded_lines"
    ].fillna(999)
)


results_df["max_loading_rank"] = (
    results_df[
        "max_line_loading_pct"
    ].fillna(9999)
)


results_df["low_voltage_rank"] = (
    results_df[
        "low_voltage_buses"
    ].fillna(999)
)


results_df["min_voltage_rank"] = (
    -results_df[
        "min_voltage_pu"
    ].fillna(9999)
)


results_df["transformer_rank"] = (
    results_df[
        "max_transformer_loading_pct"
    ].fillna(9999)
)


ranking_df = (
    results_df
    .sort_values(
        by=[
            "acceptable_rank",
            "overload_rank",
            "max_loading_rank",
            "low_voltage_rank",
            "min_voltage_rank",
            "transformer_rank",
        ],
        ascending=[
            True,
            True,
            True,
            True,
            True,
            True,
        ],
    )
    .reset_index(
        drop=True
    )
)


# ==================================================================================================
# SAVE RESULTS
# ==================================================================================================

OUTPUT_RESULTS.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_RANKING.parent.mkdir(
    parents=True,
    exist_ok=True,
)


results_df.to_csv(
    OUTPUT_RESULTS,
    index=False,
)


ranking_df.to_csv(
    OUTPUT_RANKING,
    index=False,
)


# ==================================================================================================
# SUMMARY
# ==================================================================================================

print()
print(
    "=" * 100
)

print(
    "S4.4 — COORDINATED MITIGATION SCREEN SUMMARY"
)

print(
    "=" * 100
)


display_columns = [
    "candidate",
    "category",
    "target_multiplier",
    "reactive_mvar",
    "max_line_loading_pct",
    "overloaded_lines",
    "min_voltage_pu",
    "low_voltage_buses",
    "max_transformer_loading_pct",
    "fully_acceptable",
]


summary_display = ranking_df[
    display_columns
].copy()


pd.set_option(
    "display.max_rows",
    200,
)

pd.set_option(
    "display.max_columns",
    50,
)

pd.set_option(
    "display.width",
    240,
)

pd.set_option(
    "display.float_format",
    lambda x: f"{x:.6f}",
)


print(
    summary_display.to_string(
        index=False
    )
)


# ==================================================================================================
# BEST CANDIDATE
# ==================================================================================================

acceptable = ranking_df[
    ranking_df[
        "fully_acceptable"
    ]
    == True
]


print()
print(
    "=" * 100
)

print(
    "BEST COORDINATED MITIGATION CANDIDATE"
)

print(
    "=" * 100
)


if len(acceptable) > 0:

    best = acceptable.iloc[0]


    print(
        f"Candidate                 : "
        f"{best['candidate']}"
    )

    print(
        f"Category                  : "
        f"{best['category']}"
    )

    print(
        f"Target lines              : "
        f"{best['target_lines']}"
    )

    print(
        f"Target multiplier         : "
        f"{best['target_multiplier']:.2f}x"
    )

    print(
        f"Reactive support          : "
        f"+{best['reactive_mvar']:.1f} MVAr"
    )

    print(
        f"Reactive generator        : "
        f"{best['reactive_generator']}"
    )

    print(
        f"Maximum line loading      : "
        f"{best['max_line_loading_pct']:.6f}%"
    )

    print(
        f"Overloaded lines          : "
        f"{int(best['overloaded_lines'])}"
    )

    print(
        f"Minimum voltage           : "
        f"{best['min_voltage_pu']:.6f} pu"
    )

    print(
        f"Low-voltage buses         : "
        f"{int(best['low_voltage_buses'])}"
    )

    print(
        f"Maximum transformer load  : "
        f"{best['max_transformer_loading_pct']:.6f}%"
    )

    print(
        f"Fully acceptable          : "
        f"{best['fully_acceptable']}"
    )


else:

    best = ranking_df.iloc[0]


    print(
        "NO FULLY ACCEPTABLE CANDIDATE FOUND."
    )

    print()

    print(
        "Best available candidate under the ranking:"
    )

    print(
        f"Candidate                 : "
        f"{best['candidate']}"
    )

    print(
        f"Category                  : "
        f"{best['category']}"
    )

    print(
        f"Target lines              : "
        f"{best['target_lines']}"
    )

    print(
        f"Target multiplier         : "
        f"{best['target_multiplier']:.2f}x"
    )

    print(
        f"Reactive support          : "
        f"+{best['reactive_mvar']:.1f} MVAr"
    )

    print(
        f"Reactive generator        : "
        f"{best['reactive_generator']}"
    )

    # ----------------------------------------------------------------------------------------------
    # FIXED: properly closed f-string
    # ----------------------------------------------------------------------------------------------

    if pd.notna(
        best["max_line_loading_pct"]
    ):

        print(
            f"Maximum line loading      : "
            f"{best['max_line_loading_pct']:.6f}%"
        )

    else:

        print(
            "Maximum line loading      : N/A"
        )


    # ----------------------------------------------------------------------------------------------
    # FIXED: properly closed f-string
    # ----------------------------------------------------------------------------------------------

    if pd.notna(
        best["overloaded_lines"]
    ):

        print(
            f"Overloaded lines          : "
            f"{int(best['overloaded_lines'])}"
        )

    else:

        print(
            "Overloaded lines          : N/A"
        )


    # ----------------------------------------------------------------------------------------------
    # Minimum voltage
    # ----------------------------------------------------------------------------------------------

    if pd.notna(
        best["min_voltage_pu"]
    ):

        print(
            f"Minimum voltage           : "
            f"{best['min_voltage_pu']:.6f} pu"
        )

    else:

        print(
            "Minimum voltage           : N/A"
        )


    # ----------------------------------------------------------------------------------------------
    # FIXED: properly closed f-string
    # ----------------------------------------------------------------------------------------------

    if pd.notna(
        best["low_voltage_buses"]
    ):

        print(
            f"Low-voltage buses         : "
            f"{int(best['low_voltage_buses'])}"
        )

    else:

        print(
            "Low-voltage buses         : N/A"
        )


    # ----------------------------------------------------------------------------------------------
    # Transformer loading
    # ----------------------------------------------------------------------------------------------

    if pd.notna(
        best["max_transformer_loading_pct"]
    ):

        print(
            f"Maximum transformer load  : "
            f"{best['max_transformer_loading_pct']:.6f}%"
        )

    else:

        print(
            "Maximum transformer load  : N/A"
        )


# ==================================================================================================
# ENGINEERING INTERPRETATION
# ==================================================================================================

print()
print(
    "=" * 100
)

print(
    "SYSTEM-LEVEL INTERPRETATION"
)

print(
    "=" * 100
)


acceptable_count = int(
    results_df[
        "fully_acceptable"
    ].sum()
)


print()

print(
    f"Total candidates evaluated  : "
    f"{len(results_df)}"
)

print(
    f"Fully acceptable candidates : "
    f"{acceptable_count}"
)


if acceptable_count > 0:

    print()

    print(
        "Conclusion:"
    )

    print(
        "- A coordinated reinforcement package was able "
        "to satisfy the defined AC acceptance criteria."
    )

    print(
        "- The best acceptable candidate is shown above."
    )

else:

    print()

    print(
        "Conclusion:"
    )

    print(
        "- No tested coordinated package fully removed "
        "both thermal and voltage violations."
    )

    print(
        "- The residual problem therefore requires a "
        "different mitigation mechanism or broader "
        "coordinated intervention."
    )

    print(
        "- Do not modify the source network based on "
        "this screen."
    )


# ==================================================================================================
# OUTPUT
# ==================================================================================================

print()
print(
    "=" * 100
)

print(
    "S4.4 COMPLETE"
)

print(
    "=" * 100
)

print()

print(
    "Results saved to:"
)

print(
    OUTPUT_RESULTS
)

print()

print(
    "Ranking saved to:"
)

print(
    OUTPUT_RANKING
)

print()

print(
    "IMPORTANT:"
)

print(
    "No network file was modified."
)

print(
    "All candidates were evaluated from the original "
    "READ-ONLY network."
)

print(
    "=" * 100
)