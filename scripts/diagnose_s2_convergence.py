"""
======================================================================
IRELAND GRID - S2 PEAK DEMAND AC CONVERGENCE DIAGNOSTIC
======================================================================

PURPOSE
-------
Diagnose why S2_PEAK_DEMAND succeeds or fails AC power-flow convergence
in:

1. Original optimized network
2. Second-reinforced network

IMPORTANT RULES
---------------
1. This script NEVER modifies the saved network files.
2. All PF experiments are performed on in-memory copies.
3. No reinforcement is performed.
4. AC loading is reported ONLY when the actual PyPSA PF result says
   the relevant AC sub-network converged.
5. Non-finite values are reported, never interpreted as real loading.
6. Pre-PF generation/load mismatch is reported separately from
   post-slack PF convergence.
7. PyPSA's actual pf() convergence dictionary is used whenever
   available.
8. LPF-seeded AC PF is tested separately from direct AC PF.
9. The diagnostic compares the two networks.

OUTPUTS
-------
data\\processed\\s2_convergence_diagnostic.csv
data\\processed\\s2_subnetwork_diagnostic.csv
data\\processed\\s2_bus_diagnostic.csv
data\\processed\\s2_line_diagnostic.csv
"""


from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pypsa


# =====================================================================
# CONFIGURATION
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

OPTIMIZED_FILE = DATA_DIR / "eirgrid_optimized_network.nc"
ITERATION2_FILE = DATA_DIR / "eirgrid_second_reinforced_network.nc"

SCENARIO = "S2_PEAK_DEMAND"

OUTPUT_SUMMARY = (
    DATA_DIR / "s2_convergence_diagnostic.csv"
)

OUTPUT_SUBNETWORKS = (
    DATA_DIR / "s2_subnetwork_diagnostic.csv"
)

OUTPUT_BUSES = (
    DATA_DIR / "s2_bus_diagnostic.csv"
)

OUTPUT_LINES = (
    DATA_DIR / "s2_line_diagnostic.csv"
)


# =====================================================================
# DISPLAY HELPERS
# =====================================================================

def header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def section(title):
    print()
    print("-" * 70)
    print(title)
    print("-" * 70)


# =====================================================================
# SAFE NUMERIC HELPERS
# =====================================================================

def safe_float(value):
    """
    Convert a value to finite float.

    Returns NaN for:
    - None
    - strings that cannot be converted
    - +/-inf
    - other non-finite values
    """
    try:
        value = float(value)

        if np.isfinite(value):
            return value

        return np.nan

    except Exception:
        return np.nan


def finite_array(values):
    """
    Convert values to numeric numpy array.
    """
    return pd.to_numeric(
        values,
        errors="coerce"
    ).to_numpy(dtype=float)


def all_finite(values):
    """
    True only when all supplied values are finite and there is
    at least one value.
    """
    arr = finite_array(values)

    return (
        len(arr) > 0
        and np.all(np.isfinite(arr))
    )


# =====================================================================
# LOAD NETWORK
# =====================================================================

def load_network(path, name):

    header(
        f"LOADING {name.upper()} NETWORK"
    )

    print(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Network file not found:\n{path}"
        )

    network = pypsa.Network(path)

    print("OK: Network loaded.")

    print()
    print("NETWORK")
    print(
        f"  Buses        : "
        f"{len(network.buses)}"
    )
    print(
        f"  Lines        : "
        f"{len(network.lines)}"
    )
    print(
        f"  Transformers : "
        f"{len(network.transformers)}"
    )
    print(
        f"  Links        : "
        f"{len(network.links)}"
    )
    print(
        f"  Generators   : "
        f"{len(network.generators)}"
    )
    print(
        f"  Loads        : "
        f"{len(network.loads)}"
    )

    if hasattr(network, "pypsa_version"):
        print(
            f"  PyPSA version: "
            f"{network.pypsa_version}"
        )

    return network


# =====================================================================
# SNAPSHOT CHECK
# =====================================================================

def check_snapshot(network):

    snapshots = list(network.snapshots)

    print()
    print(
        f"Snapshots ({len(snapshots)}):"
    )

    for snapshot in snapshots:
        print(
            f"  {snapshot}"
        )

    present = SCENARIO in snapshots

    print()
    print(
        f"S2 present: {present}"
    )

    return present


# =====================================================================
# GENERATOR / LOAD DISPATCH AUDIT
# =====================================================================

def dispatch_diagnostics(network, name):

    section(
        f"{name.upper()} S2 DISPATCH AUDIT"
    )

    result = {}

    # -----------------------------------------------------------------
    # Generator dispatch
    # -----------------------------------------------------------------

    generators = network.generators

    if (
        len(generators)
        and SCENARIO in network.generators_t.p_set.columns
    ):
        gen_p = network.generators_t.p_set.loc[
            SCENARIO
        ]
    else:
        gen_p = pd.Series(
            index=generators.index,
            dtype=float
        )

    total_generator_p_set = 0.0

    for gen in generators.index:

        p_set = (
            safe_float(gen_p.loc[gen])
            if gen in gen_p.index
            else np.nan
        )

        if np.isfinite(p_set):
            total_generator_p_set += p_set

        print(
            f"{gen}"
            f" | bus={generators.at[gen, 'bus']}"
            f" | carrier={generators.at[gen, 'carrier']}"
            f" | control={generators.at[gen, 'control']}"
            f" | p_nom={safe_float(generators.at[gen, 'p_nom'])}"
            f" | p_set={p_set}"
        )

    # -----------------------------------------------------------------
    # Load dispatch
    # -----------------------------------------------------------------

    loads = network.loads

    if (
        len(loads)
        and SCENARIO in network.loads_t.p_set.columns
    ):
        load_p = network.loads_t.p_set.loc[
            SCENARIO
        ]
    else:
        load_p = pd.Series(
            index=loads.index,
            dtype=float
        )

    total_load_p_set = 0.0

    for load in loads.index:

        p_set = (
            safe_float(load_p.loc[load])
            if load in load_p.index
            else np.nan
        )

        if np.isfinite(p_set):
            total_load_p_set += p_set

        print(
            f"{load}"
            f" | bus={loads.at[load, 'bus']}"
            f" | p_set={p_set}"
        )

    # -----------------------------------------------------------------
    # Balance
    # -----------------------------------------------------------------

    dispatch_balance = (
        total_generator_p_set
        - total_load_p_set
    )

    print()
    print(
        f"Total generator p_set : "
        f"{total_generator_p_set:.6f} MW"
    )

    print(
        f"Total load p_set      : "
        f"{total_load_p_set:.6f} MW"
    )

    print(
        f"Pre-PF generation-load "
        f"mismatch              : "
        f"{dispatch_balance:.6f} MW"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This is the dispatch mismatch BEFORE "
        "the AC slack solution."
    )

    print(
        "It is NOT classified as an AC convergence failure."
    )

    result[
        "generator_total_p_set"
    ] = total_generator_p_set

    result[
        "load_total_p_set"
    ] = total_load_p_set

    result[
        "pre_pf_generation_minus_load"
    ] = dispatch_balance

    return result


# =====================================================================
# STRUCTURAL DIAGNOSTICS
# =====================================================================

def structural_diagnostics(network, name):

    section(
        f"{name.upper()} STRUCTURAL DIAGNOSTICS"
    )

    result = {
        "network": name,
        "scenario": SCENARIO,
        "snapshot_count": len(network.snapshots),
        "scenario_present": (
            SCENARIO in network.snapshots
        ),
        "buses": len(network.buses),
        "lines": len(network.lines),
        "transformers": len(network.transformers),
        "links": len(network.links),
        "generators": len(network.generators),
        "loads": len(network.loads),
    }

    print(
        f"Buses        : {result['buses']}"
    )
    print(
        f"Lines        : {result['lines']}"
    )
    print(
        f"Transformers : {result['transformers']}"
    )
    print(
        f"Links        : {result['links']}"
    )
    print(
        f"Generators   : {result['generators']}"
    )
    print(
        f"Loads        : {result['loads']}"
    )

    # -----------------------------------------------------------------
    # Bus voltage
    # -----------------------------------------------------------------

    section(
        f"{name.upper()} BUS VOLTAGE STRUCTURE"
    )

    buses = network.buses

    v_nom = pd.to_numeric(
        buses["v_nom"],
        errors="coerce"
    )

    result["v_nom_min"] = safe_float(
        v_nom.min()
    )

    result["v_nom_max"] = safe_float(
        v_nom.max()
    )

    result["invalid_v_nom_buses"] = int(
        np.sum(
            ~np.isfinite(v_nom)
            | (v_nom <= 0)
        )
    )

    print(
        f"Minimum v_nom : "
        f"{result['v_nom_min']}"
    )

    print(
        f"Maximum v_nom : "
        f"{result['v_nom_max']}"
    )

    print(
        f"Invalid v_nom buses : "
        f"{result['invalid_v_nom_buses']}"
    )

    # -----------------------------------------------------------------
    # Line impedance
    # -----------------------------------------------------------------

    section(
        f"{name.upper()} LINE PARAMETER STRUCTURE"
    )

    lines = network.lines

    if len(lines):

        r = pd.to_numeric(
            lines["r"],
            errors="coerce"
        )

        x = pd.to_numeric(
            lines["x"],
            errors="coerce"
        )

        s_nom = pd.to_numeric(
            lines["s_nom"],
            errors="coerce"
        )

        result["nonfinite_r"] = int(
            np.sum(~np.isfinite(r))
        )

        result["nonfinite_x"] = int(
            np.sum(~np.isfinite(x))
        )

        result["zero_r"] = int(
            np.sum(r == 0)
        )

        result["zero_x"] = int(
            np.sum(x == 0)
        )

        result["nonpositive_s_nom"] = int(
            np.sum(
                ~np.isfinite(s_nom)
                | (s_nom <= 0)
            )
        )

        nonzero_finite_x = np.abs(
            x[
                np.isfinite(x)
                & (x != 0)
            ]
        )

        if len(nonzero_finite_x):

            result["minimum_abs_x"] = safe_float(
                nonzero_finite_x.min()
            )

            result["maximum_abs_x"] = safe_float(
                nonzero_finite_x.max()
            )

        else:

            result["minimum_abs_x"] = np.nan
            result["maximum_abs_x"] = np.nan

    else:

        result["nonfinite_r"] = 0
        result["nonfinite_x"] = 0
        result["zero_r"] = 0
        result["zero_x"] = 0
        result["nonpositive_s_nom"] = 0
        result["minimum_abs_x"] = np.nan
        result["maximum_abs_x"] = np.nan

    print(
        f"Non-finite r : "
        f"{result['nonfinite_r']}"
    )

    print(
        f"Non-finite x : "
        f"{result['nonfinite_x']}"
    )

    print(
        f"Zero r       : "
        f"{result['zero_r']}"
    )

    print(
        f"Zero x       : "
        f"{result['zero_x']}"
    )

    print(
        f"Non-positive s_nom : "
        f"{result['nonpositive_s_nom']}"
    )

    print(
        f"Minimum |x| : "
        f"{result['minimum_abs_x']}"
    )

    # -----------------------------------------------------------------
    # Transformers
    # -----------------------------------------------------------------

    section(
        f"{name.upper()} TRANSFORMER STRUCTURE"
    )

    transformers = network.transformers

    result[
        "transformer_nonfinite_r"
    ] = 0

    result[
        "transformer_nonfinite_x"
    ] = 0

    result[
        "transformer_zero_x"
    ] = 0

    if len(transformers):

        tr_r = pd.to_numeric(
            transformers["r"],
            errors="coerce"
        )

        tr_x = pd.to_numeric(
            transformers["x"],
            errors="coerce"
        )

        result[
            "transformer_nonfinite_r"
        ] = int(
            np.sum(~np.isfinite(tr_r))
        )

        result[
            "transformer_nonfinite_x"
        ] = int(
            np.sum(~np.isfinite(tr_x))
        )

        result[
            "transformer_zero_x"
        ] = int(
            np.sum(tr_x == 0)
        )

        print(
            f"Transformers : "
            f"{len(transformers)}"
        )

        print(
            f"Non-finite transformer r : "
            f"{result['transformer_nonfinite_r']}"
        )

        print(
            f"Non-finite transformer x : "
            f"{result['transformer_nonfinite_x']}"
        )

        print(
            f"Zero transformer x : "
            f"{result['transformer_zero_x']}"
        )

    else:

        print(
            "No transformers."
        )

    return result


# =====================================================================
# BUS DIAGNOSTIC
# =====================================================================

def build_bus_diagnostic(network, name):

    section(
        f"{name.upper()} BUS POWER DIAGNOSTIC"
    )

    buses = network.buses

    gen_by_bus = pd.Series(
        0.0,
        index=buses.index,
        dtype=float
    )

    load_by_bus = pd.Series(
        0.0,
        index=buses.index,
        dtype=float
    )

    # -----------------------------------------------------------------
    # Generators
    # -----------------------------------------------------------------

    if (
        len(network.generators)
        and SCENARIO
        in network.generators_t.p_set.columns
    ):

        gen_values = (
            network.generators_t.p_set
            .loc[SCENARIO]
        )

        for gen in network.generators.index:

            bus = network.generators.at[
                gen,
                "bus"
            ]

            if gen not in gen_values.index:
                continue

            value = safe_float(
                gen_values.loc[gen]
            )

            if (
                np.isfinite(value)
                and bus in gen_by_bus.index
            ):
                gen_by_bus.loc[bus] += value

    # -----------------------------------------------------------------
    # Loads
    # -----------------------------------------------------------------

    if (
        len(network.loads)
        and SCENARIO
        in network.loads_t.p_set.columns
    ):

        load_values = (
            network.loads_t.p_set
            .loc[SCENARIO]
        )

        for load in network.loads.index:

            bus = network.loads.at[
                load,
                "bus"
            ]

            if load not in load_values.index:
                continue

            value = safe_float(
                load_values.loc[load]
            )

            if (
                np.isfinite(value)
                and bus in load_by_bus.index
            ):
                load_by_bus.loc[bus] += value

    # -----------------------------------------------------------------
    # Rows
    # -----------------------------------------------------------------

    rows = []

    for bus in buses.index:

        generation = safe_float(
            gen_by_bus.loc[bus]
        )

        load = safe_float(
            load_by_bus.loc[bus]
        )

        rows.append({
            "network": name,
            "scenario": SCENARIO,
            "bus": bus,
            "v_nom": safe_float(
                buses.at[bus, "v_nom"]
            ),
            "generator_p_set": generation,
            "load_p_set": load,
            "net_injection": safe_float(
                generation - load
            ),
        })

    df = pd.DataFrame(rows)

    print()

    print(
        df[
            [
                "bus",
                "generator_p_set",
                "load_p_set",
                "net_injection",
            ]
        ]
        .sort_values(
            "net_injection"
        )
        .to_string(
            index=False
        )
    )

    return df


# =====================================================================
# LINE STRUCTURAL DIAGNOSTIC
# =====================================================================

def build_line_diagnostic(network, name):

    section(
        f"{name.upper()} LINE STRUCTURAL DIAGNOSTIC"
    )

    lines = network.lines

    rows = []

    for line in lines.index:

        bus0 = lines.at[
            line,
            "bus0"
        ]

        bus1 = lines.at[
            line,
            "bus1"
        ]

        r = safe_float(
            lines.at[line, "r"]
        )

        x = safe_float(
            lines.at[line, "x"]
        )

        s_nom = safe_float(
            lines.at[line, "s_nom"]
        )

        abs_x = (
            abs(x)
            if np.isfinite(x)
            else np.nan
        )

        r_over_x = (
            r / x
            if (
                np.isfinite(r)
                and np.isfinite(x)
                and x != 0
            )
            else np.nan
        )

        rows.append({
            "network": name,
            "scenario": SCENARIO,
            "line": line,
            "bus0": bus0,
            "bus1": bus1,
            "r": r,
            "x": x,
            "s_nom": s_nom,
            "abs_x": abs_x,
            "r_over_x": r_over_x,
        })

    df = pd.DataFrame(rows)

    if len(df):

        print()
        print(
            "LOWEST |X| LINES"
        )

        print(
            df.sort_values(
                "abs_x",
                na_position="last"
            )
            .head(15)
            .to_string(
                index=False
            )
        )

    return df


# =====================================================================
# RUN ACTUAL PYPSA AC POWER FLOW
# =====================================================================

def run_single_ac_pf(
    network,
    name,
    method,
    use_seed=False,
    distribute_slack=False,
):

    """
    Run one AC PF experiment on a COPY of the network.

    method:
        direct
        lpf_seeded

    Returns:
        summary dictionary
        subnetwork DataFrame
        line DataFrame
    """

    section(
        f"{name.upper()} AC PF - {method.upper()}"
    )

    test_network = network.copy()

    result = {
        "network": name,
        "scenario": SCENARIO,
        "pf_method": method,
        "pf_use_seed": use_seed,
        "pf_distribute_slack": distribute_slack,
        "pf_exception": "",
        "pf_return_available": False,
        "pf_converged": False,
        "pf_all_subnetworks_converged": False,
        "pf_max_error": np.nan,
        "pf_max_iterations": np.nan,
        "loading_valid": False,
        "maximum_line_loading_pct": np.nan,
        "overloaded_lines": np.nan,
        "minimum_voltage_pu": np.nan,
        "maximum_voltage_pu": np.nan,
        "voltage_values_finite": False,
        "line_flow_values_finite": False,
    }

    empty_subnetwork = pd.DataFrame()

    empty_lines = pd.DataFrame()

    if SCENARIO not in test_network.snapshots:

        result["pf_exception"] = (
            "S2_PEAK_DEMAND missing"
        )

        print(
            "S2 missing. PF skipped."
        )

        return (
            result,
            empty_subnetwork,
            empty_lines,
        )

    # -----------------------------------------------------------------
    # LPF seed
    # -----------------------------------------------------------------

    if use_seed:

        print()
        print(
            "STEP 1: Running LPF for Newton-Raphson seed..."
        )

        try:

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                test_network.lpf(
                    snapshots=[SCENARIO]
                )

            print(
                "LPF seed: SUCCESS"
            )

        except Exception as exc:

            print()
            print(
                "LPF SEED FAILED"
            )

            print(
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            result["pf_exception"] = (
                f"LPF seed failed: "
                f"{type(exc).__name__}: {exc}"
            )

            return (
                result,
                empty_subnetwork,
                empty_lines,
            )

    # -----------------------------------------------------------------
    # AC PF
    # -----------------------------------------------------------------

    print()
    print(
        "STEP 2: Running nonlinear AC power flow..."
    )

    pf_result = None

    try:

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            pf_result = test_network.pf(
                snapshots=[SCENARIO],
                use_seed=use_seed,
                distribute_slack=distribute_slack,
            )

        result[
            "pf_return_available"
        ] = isinstance(
            pf_result,
            dict
        )

        print()
        print(
            "PyPSA pf() returned:"
        )

        print(
            pf_result
        )

    except Exception as exc:

        print()
        print(
            "AC POWER FLOW EXCEPTION"
        )

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        result["pf_exception"] = (
            f"{type(exc).__name__}: {exc}"
        )

        return (
            result,
            empty_subnetwork,
            empty_lines,
        )

    # -----------------------------------------------------------------
    # Extract actual PyPSA convergence result
    # -----------------------------------------------------------------

    subnetwork_rows = []

    if isinstance(pf_result, dict):

        converged_table = (
            pf_result.get(
                "converged"
            )
        )

        error_table = (
            pf_result.get(
                "error"
            )
        )

        iteration_table = (
            pf_result.get(
                "n_iter"
            )
        )

        if converged_table is not None:

            converged_series = (
                converged_table.loc[
                    SCENARIO
                ]
                if SCENARIO
                in converged_table.index
                else pd.Series(
                    dtype=bool
                )
            )

            if error_table is not None:
                error_series = (
                    error_table.loc[
                        SCENARIO
                    ]
                    if SCENARIO
                    in error_table.index
                    else pd.Series(
                        dtype=float
                    )
                )
            else:
                error_series = pd.Series(
                    dtype=float
                )

            if iteration_table is not None:
                iteration_series = (
                    iteration_table.loc[
                        SCENARIO
                    ]
                    if SCENARIO
                    in iteration_table.index
                    else pd.Series(
                        dtype=float
                    )
                )
            else:
                iteration_series = pd.Series(
                    dtype=float
                )

            for subnetwork in converged_series.index:

                converged_value = (
                    bool(
                        converged_series.loc[
                            subnetwork
                        ]
                    )
                    if pd.notna(
                        converged_series.loc[
                            subnetwork
                        ]
                    )
                    else False
                )

                error_value = (
                    safe_float(
                        error_series.loc[
                            subnetwork
                        ]
                    )
                    if subnetwork
                    in error_series.index
                    else np.nan
                )

                iteration_value = (
                    safe_float(
                        iteration_series.loc[
                            subnetwork
                        ]
                    )
                    if subnetwork
                    in iteration_series.index
                    else np.nan
                )

                subnetwork_rows.append({
                    "network": name,
                    "scenario": SCENARIO,
                    "pf_method": method,
                    "subnetwork": subnetwork,
                    "converged": converged_value,
                    "error": error_value,
                    "n_iter": iteration_value,
                })

    # -----------------------------------------------------------------
    # Fallback for old PyPSA versions
    # -----------------------------------------------------------------

    if not subnetwork_rows:

        print()
        print(
            "WARNING: PyPSA did not expose a convergence table."
        )

        print(
            "Numerical state will be reported separately."
        )

    subnetwork_df = pd.DataFrame(
        subnetwork_rows
    )

    # -----------------------------------------------------------------
    # Actual convergence status
    # -----------------------------------------------------------------

    if len(subnetwork_df):

        result[
            "pf_all_subnetworks_converged"
        ] = bool(
            subnetwork_df[
                "converged"
            ].all()
        )

        result[
            "pf_converged"
        ] = result[
            "pf_all_subnetworks_converged"
        ]

        finite_errors = pd.to_numeric(
            subnetwork_df["error"],
            errors="coerce"
        )

        if np.any(
            np.isfinite(
                finite_errors
            )
        ):

            result[
                "pf_max_error"
            ] = safe_float(
                np.nanmax(
                    finite_errors
                )
            )

        finite_iterations = pd.to_numeric(
            subnetwork_df["n_iter"],
            errors="coerce"
        )

        if np.any(
            np.isfinite(
                finite_iterations
            )
        ):

            result[
                "pf_max_iterations"
            ] = safe_float(
                np.nanmax(
                    finite_iterations
                )
            )

    # -----------------------------------------------------------------
    # Voltage diagnostic
    # -----------------------------------------------------------------

    voltage = None

    if (
        hasattr(
            test_network,
            "buses_t"
        )
        and hasattr(
            test_network.buses_t,
            "v_mag_pu"
        )
        and SCENARIO
        in test_network.buses_t.v_mag_pu.index
    ):

        voltage = (
            test_network
            .buses_t
            .v_mag_pu
            .loc[SCENARIO]
        )

        voltage_values = finite_array(
            voltage
        )

        result[
            "voltage_values_finite"
        ] = (
            len(voltage_values) > 0
            and np.all(
                np.isfinite(
                    voltage_values
                )
            )
        )

        finite_voltage = (
            voltage_values[
                np.isfinite(
                    voltage_values
                )
            ]
        )

        if len(finite_voltage):

            result[
                "minimum_voltage_pu"
            ] = float(
                np.min(
                    finite_voltage
                )
            )

            result[
                "maximum_voltage_pu"
            ] = float(
                np.max(
                    finite_voltage
                )
            )

    # -----------------------------------------------------------------
    # AC line flows
    # -----------------------------------------------------------------

    line_rows = []

    if len(test_network.lines):

        lines = test_network.lines

        p0 = None
        p1 = None
        s0 = None
        s1 = None

        if (
            hasattr(
                test_network.lines_t,
                "p0"
            )
            and SCENARIO
            in test_network.lines_t.p0.index
        ):

            p0 = (
                test_network
                .lines_t
                .p0
                .loc[SCENARIO]
            )

        if (
            hasattr(
                test_network.lines_t,
                "p1"
            )
            and SCENARIO
            in test_network.lines_t.p1.index
        ):

            p1 = (
                test_network
                .lines_t
                .p1
                .loc[SCENARIO]
            )

        if (
            hasattr(
                test_network.lines_t,
                "s0"
            )
            and SCENARIO
            in test_network.lines_t.s0.index
        ):

            s0 = (
                test_network
                .lines_t
                .s0
                .loc[SCENARIO]
            )

        if (
            hasattr(
                test_network.lines_t,
                "s1"
            )
            and SCENARIO
            in test_network.lines_t.s1.index
        ):

            s1 = (
                test_network
                .lines_t
                .s1
                .loc[SCENARIO]
            )

        # -------------------------------------------------------------
        # Prefer apparent-power loading if available.
        # Otherwise use active power as fallback.
        # -------------------------------------------------------------

        for line in lines.index:

            line_s_nom = safe_float(
                lines.at[
                    line,
                    "s_nom"
                ]
            )

            line_p0 = (
                safe_float(
                    p0.loc[line]
                )
                if (
                    p0 is not None
                    and line in p0.index
                )
                else np.nan
            )

            line_p1 = (
                safe_float(
                    p1.loc[line]
                )
                if (
                    p1 is not None
                    and line in p1.index
                )
                else np.nan
            )

            line_s0 = (
                safe_float(
                    s0.loc[line]
                )
                if (
                    s0 is not None
                    and line in s0.index
                )
                else np.nan
            )

            line_s1 = (
                safe_float(
                    s1.loc[line]
                )
                if (
                    s1 is not None
                    and line in s1.index
                )
                else np.nan
            )

            # ---------------------------------------------------------
            # Determine loading from AC result
            # ---------------------------------------------------------

            loading0 = np.nan
            loading1 = np.nan

            if (
                np.isfinite(line_s_nom)
                and line_s_nom > 0
            ):

                if np.isfinite(line_s0):

                    loading0 = (
                        abs(line_s0)
                        / line_s_nom
                        * 100.0
                    )

                elif np.isfinite(line_p0):

                    loading0 = (
                        abs(line_p0)
                        / line_s_nom
                        * 100.0
                    )

                if np.isfinite(line_s1):

                    loading1 = (
                        abs(line_s1)
                        / line_s_nom
                        * 100.0
                    )

                elif np.isfinite(line_p1):

                    loading1 = (
                        abs(line_p1)
                        / line_s_nom
                        * 100.0
                    )

            max_loading = np.nan

            finite_loading = [
                value
                for value in [
                    loading0,
                    loading1
                ]
                if np.isfinite(value)
            ]

            if finite_loading:

                max_loading = max(
                    finite_loading
                )

            line_rows.append({
                "network": name,
                "scenario": SCENARIO,
                "pf_method": method,
                "line": line,
                "bus0": lines.at[
                    line,
                    "bus0"
                ],
                "bus1": lines.at[
                    line,
                    "bus1"
                ],
                "s_nom": line_s_nom,
                "p0": line_p0,
                "p1": line_p1,
                "s0": line_s0,
                "s1": line_s1,
                "loading0_pct": loading0,
                "loading1_pct": loading1,
                "maximum_loading_pct": max_loading,
            })

        line_df = pd.DataFrame(
            line_rows
        )

    else:

        line_df = pd.DataFrame()

    # -----------------------------------------------------------------
    # Flow finite state
    # -----------------------------------------------------------------

    if len(line_df):

        flow_columns = [
            "p0",
            "p1",
        ]

        available_flow_columns = [
            column
            for column in flow_columns
            if column in line_df.columns
        ]

        flow_values = line_df[
            available_flow_columns
        ].to_numpy(
            dtype=float
        )

        result[
            "line_flow_values_finite"
        ] = (
            flow_values.size > 0
            and np.all(
                np.isfinite(
                    flow_values
                )
            )
        )

    # -----------------------------------------------------------------
    # VALID LOADING RULE
    # -----------------------------------------------------------------

    # Loading is accepted ONLY when the actual PyPSA convergence
    # table says all subnetworks converged.
    #
    # We deliberately do NOT infer convergence from finite values alone.

    if (
        result[
            "pf_all_subnetworks_converged"
        ]
        and len(line_df)
    ):

        loading_values = pd.concat(
            [
                pd.to_numeric(
                    line_df[
                        "loading0_pct"
                    ],
                    errors="coerce"
                ),
                pd.to_numeric(
                    line_df[
                        "loading1_pct"
                    ],
                    errors="coerce"
                ),
            ],
            ignore_index=True
        )

        finite_loading = (
            loading_values[
                np.isfinite(
                    loading_values
                )
            ]
        )

        if len(finite_loading):

            result[
                "loading_valid"
            ] = True

            result[
                "maximum_line_loading_pct"
            ] = float(
                np.max(
                    finite_loading
                )
            )

            result[
                "overloaded_lines"
            ] = int(
                np.sum(
                    line_df[
                        "maximum_loading_pct"
                    ] > 100.0
                )
            )

    # -----------------------------------------------------------------
    # Print final AC state
    # -----------------------------------------------------------------

    print()
    print(
        "ACTUAL PYPSA AC CONVERGENCE"
    )

    print(
        f"  PF returned result : "
        f"{result['pf_return_available']}"
    )

    print(
        f"  All subnetworks converged : "
        f"{result['pf_all_subnetworks_converged']}"
    )

    print(
        f"  Maximum PF error : "
        f"{result['pf_max_error']}"
    )

    print(
        f"  Maximum iterations : "
        f"{result['pf_max_iterations']}"
    )

    print()
    print(
        "VOLTAGE STATE"
    )

    print(
        f"  Finite : "
        f"{result['voltage_values_finite']}"
    )

    print(
        f"  Minimum V : "
        f"{result['minimum_voltage_pu']}"
    )

    print(
        f"  Maximum V : "
        f"{result['maximum_voltage_pu']}"
    )

    print()
    print(
        "LOADING STATE"
    )

    if result["loading_valid"]:

        print(
            "  VALID"
        )

        print(
            f"  Maximum loading : "
            f"{result['maximum_line_loading_pct']:.6f}%"
        )

        print(
            f"  Lines > 100% : "
            f"{result['overloaded_lines']}"
        )

    else:

        print(
            "  REJECTED"
        )

        print(
            "  Reason: AC convergence was not "
            "established for all subnetworks."
        )

    return (
        result,
        subnetwork_df,
        line_df,
    )


# =====================================================================
# AC TEST SUITE
# =====================================================================

def run_ac_test_suite(network, name):

    section(
        f"{name.upper()} AC CONVERGENCE TEST SUITE"
    )

    # ---------------------------------------------------------------
    # TEST A: Direct Newton-Raphson
    # ---------------------------------------------------------------

    direct_result, direct_subnets, direct_lines = (
        run_single_ac_pf(
            network=network,
            name=name,
            method="direct",
            use_seed=False,
            distribute_slack=False,
        )
    )

    # ---------------------------------------------------------------
    # TEST B: LPF-seeded Newton-Raphson
    # ---------------------------------------------------------------

    seeded_result, seeded_subnets, seeded_lines = (
        run_single_ac_pf(
            network=network,
            name=name,
            method="lpf_seeded",
            use_seed=True,
            distribute_slack=False,
        )
    )

    # ---------------------------------------------------------------
    # Test C only if both previous methods fail
    #
    # Distributed slack is diagnostically useful, but it changes the
    # slack allocation assumption. Therefore it is NOT used as the
    # primary convergence result.
    # ---------------------------------------------------------------

    distributed_result = None
    distributed_subnets = pd.DataFrame()
    distributed_lines = pd.DataFrame()

    if not (
        direct_result["pf_converged"]
        or seeded_result["pf_converged"]
    ):

        print()
        print(
            "Both primary AC attempts failed."
        )

        print(
            "Running distributed-slack diagnostic "
            "as a separate experiment."
        )

        (
            distributed_result,
            distributed_subnets,
            distributed_lines,
        ) = run_single_ac_pf(
            network=network,
            name=name,
            method="distributed_slack",
            use_seed=True,
            distribute_slack=True,
        )

    # ---------------------------------------------------------------
    # Choose primary result
    # ---------------------------------------------------------------

    if direct_result["pf_converged"]:

        primary = direct_result
        primary_subnets = direct_subnets
        primary_lines = direct_lines

        primary_method = "direct"

    elif seeded_result["pf_converged"]:

        primary = seeded_result
        primary_subnets = seeded_subnets
        primary_lines = seeded_lines

        primary_method = "lpf_seeded"

    else:

        primary = direct_result.copy()

        primary["primary_result"] = False

        primary_method = "none"

        primary_subnets = direct_subnets
        primary_lines = direct_lines

    # ---------------------------------------------------------------
    # Add suite-level results
    # ---------------------------------------------------------------

    suite_summary = {
        "network": name,
        "scenario": SCENARIO,

        "direct_pf_converged":
            direct_result[
                "pf_converged"
            ],

        "direct_pf_max_error":
            direct_result[
                "pf_max_error"
            ],

        "direct_pf_max_iterations":
            direct_result[
                "pf_max_iterations"
            ],

        "direct_loading_valid":
            direct_result[
                "loading_valid"
            ],

        "seeded_pf_converged":
            seeded_result[
                "pf_converged"
            ],

        "seeded_pf_max_error":
            seeded_result[
                "pf_max_error"
            ],

        "seeded_pf_max_iterations":
            seeded_result[
                "pf_max_iterations"
            ],

        "seeded_loading_valid":
            seeded_result[
                "loading_valid"
            ],

        "primary_pf_method":
            primary_method,

        "primary_pf_converged":
            primary.get(
                "pf_converged",
                False
            ),

        "primary_loading_valid":
            primary.get(
                "loading_valid",
                False
            ),

        "primary_max_loading_pct":
            primary.get(
                "maximum_line_loading_pct",
                np.nan
            ),

        "primary_overloaded_lines":
            primary.get(
                "overloaded_lines",
                np.nan
            ),

        "primary_min_voltage_pu":
            primary.get(
                "minimum_voltage_pu",
                np.nan
            ),

        "primary_max_voltage_pu":
            primary.get(
                "maximum_voltage_pu",
                np.nan
            ),
    }

    # ---------------------------------------------------------------
    # Distributed slack
    # ---------------------------------------------------------------

    if distributed_result is not None:

        suite_summary[
            "distributed_slack_pf_converged"
        ] = distributed_result[
            "pf_converged"
        ]

        suite_summary[
            "distributed_slack_loading_valid"
        ] = distributed_result[
            "loading_valid"
        ]

        suite_summary[
            "distributed_slack_max_error"
        ] = distributed_result[
            "pf_max_error"
        ]

        suite_summary[
            "distributed_slack_max_loading_pct"
        ] = distributed_result[
            "maximum_line_loading_pct"
        ]

    else:

        suite_summary[
            "distributed_slack_pf_converged"
        ] = np.nan

        suite_summary[
            "distributed_slack_loading_valid"
        ] = np.nan

        suite_summary[
            "distributed_slack_max_error"
        ] = np.nan

        suite_summary[
            "distributed_slack_max_loading_pct"
        ] = np.nan

    # ---------------------------------------------------------------
    # Combine subnetwork diagnostics
    # ---------------------------------------------------------------

    subnetwork_frames = [
        frame
        for frame in [
            direct_subnets,
            seeded_subnets,
            distributed_subnets,
        ]
        if len(frame)
    ]

    if subnetwork_frames:

        combined_subnets = pd.concat(
            subnetwork_frames,
            ignore_index=True
        )

    else:

        combined_subnets = pd.DataFrame()

    # ---------------------------------------------------------------
    # Combine line diagnostics
    # ---------------------------------------------------------------

    line_frames = [
        frame
        for frame in [
            direct_lines,
            seeded_lines,
            distributed_lines,
        ]
        if len(frame)
    ]

    if line_frames:

        combined_lines = pd.concat(
            line_frames,
            ignore_index=True
        )

    else:

        combined_lines = pd.DataFrame()

    return (
        suite_summary,
        combined_subnets,
        combined_lines,
    )


# =====================================================================
# NETWORK COMPARISON
# =====================================================================

def compare_networks(
    optimized,
    iteration2
):

    section(
        "ORIGINAL VS SECOND-REINFORCED COMPARISON"
    )

    metrics = [
        (
            "Buses",
            "buses"
        ),
        (
            "Lines",
            "lines"
        ),
        (
            "Transformers",
            "transformers"
        ),
        (
            "Generators",
            "generators"
        ),
        (
            "Loads",
            "loads"
        ),
        (
            "Generator p_set",
            "generator_total_p_set"
        ),
        (
            "Load p_set",
            "load_total_p_set"
        ),
        (
            "Pre-PF generation-load mismatch",
            "pre_pf_generation_minus_load"
        ),
        (
            "Minimum |X|",
            "minimum_abs_x"
        ),
        (
            "Zero-X lines",
            "zero_x"
        ),
        (
            "Non-finite X",
            "nonfinite_x"
        ),
        (
            "Non-positive s_nom",
            "nonpositive_s_nom"
        ),
        (
            "Primary PF converged",
            "primary_pf_converged"
        ),
        (
            "Primary loading valid",
            "primary_loading_valid"
        ),
        (
            "Primary maximum loading %",
            "primary_max_loading_pct"
        ),
        (
            "Primary overloaded lines",
            "primary_overloaded_lines"
        ),
        (
            "Primary minimum V pu",
            "primary_min_voltage_pu"
        ),
    ]

    print()

    print(
        f"{'Metric':40s}"
        f"{'Optimized':>18s}"
        f"{'Iteration-2':>18s}"
    )

    print(
        "-" * 76
    )

    for label, key in metrics:

        a = optimized.get(
            key,
            np.nan
        )

        b = iteration2.get(
            key,
            np.nan
        )

        print(
            f"{label:40s}"
            f"{str(a):>18s}"
            f"{str(b):>18s}"
        )


# =====================================================================
# FINAL INTERPRETATION
# =====================================================================

def final_interpretation(
    optimized,
    iteration2,
):

    header(
        "DIAGNOSTIC INTERPRETATION"
    )

    print()
    print(
        "1. PRE-PF DISPATCH BALANCE"
    )

    print(
        f"   Optimized   : "
        f"{optimized.get('pre_pf_generation_minus_load')}"
    )

    print(
        f"   Iteration-2 : "
        f"{iteration2.get('pre_pf_generation_minus_load')}"
    )

    print()
    print(
        "This mismatch is NOT automatically a PF failure."
    )

    print(
        "The AC power flow may adjust the designated slack "
        "generator to balance the system."
    )

    print()
    print(
        "2. ACTUAL PYPSA AC CONVERGENCE"
    )

    print(
        f"   Optimized   : "
        f"{optimized.get('primary_pf_converged')}"
    )

    print(
        f"   Iteration-2 : "
        f"{iteration2.get('primary_pf_converged')}"
    )

    print()
    print(
        "3. PRIMARY PF METHOD"
    )

    print(
        f"   Optimized   : "
        f"{optimized.get('primary_pf_method')}"
    )

    print(
        f"   Iteration-2 : "
        f"{iteration2.get('primary_pf_method')}"
    )

    print()
    print(
        "4. LOADING VALIDITY"
    )

    print(
        f"   Optimized   : "
        f"{optimized.get('primary_loading_valid')}"
    )

    print(
        f"   Iteration-2 : "
        f"{iteration2.get('primary_loading_valid')}"
    )

    print()
    print(
        "5. IMPORTANT DECISION RULE"
    )

    print(
        "If primary_pf_converged = False, "
        "do NOT use any line-loading value from that "
        "failed solve to claim a bottleneck."
    )

    print(
        "If primary_pf_converged = True, "
        "the corresponding AC line-loading results are valid "
        "for diagnostic interpretation."
    )

    print()
    print(
        "6. REINFORCEMENT INTERPRETATION"
    )

    opt_conv = optimized.get(
        "primary_pf_converged",
        False
    )

    it2_conv = iteration2.get(
        "primary_pf_converged",
        False
    )

    if (
        not opt_conv
        and not it2_conv
    ):

        print(
            "Both networks still fail the primary AC "
            "convergence criterion."
        )

        print(
            "Therefore reinforcement has NOT yet demonstrated "
            "a valid AC operating solution for S2."
        )

    elif (
        not opt_conv
        and it2_conv
    ):

        print(
            "The second-reinforced network converges while "
            "the original network does not."
        )

        print(
            "This is evidence that the network modification "
            "changed the AC solvability state."
        )

        print(
            "However, engineering interpretation should "
            "still examine voltage, reactive power, and "
            "line loading."
        )

    elif (
        opt_conv
        and not it2_conv
    ):

        print(
            "The original network converges but the "
            "second-reinforced network does not."
        )

        print(
            "This requires investigation because the "
            "reinforcement changed the numerical/physical "
            "AC solution state adversely."
        )

    else:

        print(
            "Both networks have a valid primary AC solution."
        )

        print(
            "The next comparison should focus on:"
        )

        print(
            "  - maximum AC apparent-power loading"
        )

        print(
            "  - minimum bus voltage"
        )

        print(
            "  - maximum bus voltage"
        )

        print(
            "  - reactive-power dispatch"
        )

        print(
            "  - line losses"
        )


# =====================================================================
# MAIN
# =====================================================================

def main():

    header(
        "IRELAND GRID - S2 PEAK DEMAND "
        "AC CONVERGENCE DIAGNOSTIC"
    )

    print()
    print(
        "PURPOSE"
    )

    print(
        "Diagnose S2_PEAK_DEMAND AC convergence "
        "before making further reinforcement decisions."
    )

    print()
    print(
        "SAFETY RULES"
    )

    print(
        "  - Saved networks are never modified."
    )

    print(
        "  - PF experiments use network copies."
    )

    print(
        "  - No reinforcement is performed."
    )

    print(
        "  - Failed AC PF is not converted into overload."
    )

    print(
        "  - Actual PyPSA convergence results are used."
    )

    # -----------------------------------------------------------------
    # Load networks
    # -----------------------------------------------------------------

    optimized = load_network(
        OPTIMIZED_FILE,
        "optimized"
    )

    iteration2 = load_network(
        ITERATION2_FILE,
        "iteration-2"
    )

    # -----------------------------------------------------------------
    # Snapshot validation
    # -----------------------------------------------------------------

    section(
        "SCENARIO VALIDATION"
    )

    optimized_s2 = check_snapshot(
        optimized
    )

    iteration2_s2 = check_snapshot(
        iteration2
    )

    if not optimized_s2:
        raise RuntimeError(
            "S2_PEAK_DEMAND is missing from "
            "the optimized network."
        )

    if not iteration2_s2:
        raise RuntimeError(
            "S2_PEAK_DEMAND is missing from "
            "the second-reinforced network."
        )

    # -----------------------------------------------------------------
    # Structural diagnostics
    # -----------------------------------------------------------------

    optimized_summary = structural_diagnostics(
        optimized,
        "optimized"
    )

    iteration2_summary = structural_diagnostics(
        iteration2,
        "iteration-2"
    )

    # -----------------------------------------------------------------
    # Dispatch diagnostics
    # -----------------------------------------------------------------

    optimized_dispatch = dispatch_diagnostics(
        optimized,
        "optimized"
    )

    iteration2_dispatch = dispatch_diagnostics(
        iteration2,
        "iteration-2"
    )

    optimized_summary.update(
        optimized_dispatch
    )

    iteration2_summary.update(
        iteration2_dispatch
    )

    # -----------------------------------------------------------------
    # Bus diagnostics
    # -----------------------------------------------------------------

    optimized_buses = build_bus_diagnostic(
        optimized,
        "optimized"
    )

    iteration2_buses = build_bus_diagnostic(
        iteration2,
        "iteration-2"
    )

    bus_combined = pd.concat(
        [
            optimized_buses,
            iteration2_buses,
        ],
        ignore_index=True
    )

    # -----------------------------------------------------------------
    # Line structural diagnostics
    # -----------------------------------------------------------------

    optimized_lines_structural = (
        build_line_diagnostic(
            optimized,
            "optimized"
        )
    )

    iteration2_lines_structural = (
        build_line_diagnostic(
            iteration2,
            "iteration-2"
        )
    )

    # -----------------------------------------------------------------
    # AC PF suite
    # -----------------------------------------------------------------

    (
        optimized_ac,
        optimized_subnets,
        optimized_pf_lines,
    ) = run_ac_test_suite(
        optimized,
        "optimized"
    )

    (
        iteration2_ac,
        iteration2_subnets,
        iteration2_pf_lines,
    ) = run_ac_test_suite(
        iteration2,
        "iteration-2"
    )

    # -----------------------------------------------------------------
    # Merge AC summaries
    # -----------------------------------------------------------------

    optimized_summary.update(
        optimized_ac
    )

    iteration2_summary.update(
        iteration2_ac
    )

    # -----------------------------------------------------------------
    # Combine subnetwork results
    # -----------------------------------------------------------------

    subnetwork_frames = [
        frame
        for frame in [
            optimized_subnets,
            iteration2_subnets,
        ]
        if len(frame)
    ]

    if subnetwork_frames:

        subnetwork_combined = pd.concat(
            subnetwork_frames,
            ignore_index=True
        )

    else:

        subnetwork_combined = pd.DataFrame()

    # -----------------------------------------------------------------
    # Combine line results
    # -----------------------------------------------------------------

    pf_line_frames = [
        frame
        for frame in [
            optimized_pf_lines,
            iteration2_pf_lines,
        ]
        if len(frame)
    ]

    if pf_line_frames:

        pf_line_combined = pd.concat(
            pf_line_frames,
            ignore_index=True
        )

    else:

        pf_line_combined = pd.DataFrame()

    # -----------------------------------------------------------------
    # Combine structural and PF line diagnostics
    # -----------------------------------------------------------------

    structural_frames = [
        optimized_lines_structural,
        iteration2_lines_structural,
    ]

    structural_combined = pd.concat(
        structural_frames,
        ignore_index=True
    )

    if len(pf_line_combined):

        # Keep structural information and PF information separate
        # through a keyed merge.

        line_combined = structural_combined.merge(
            pf_line_combined[
                [
                    "network",
                    "scenario",
                    "pf_method",
                    "line",
                    "p0",
                    "p1",
                    "s0",
                    "s1",
                    "loading0_pct",
                    "loading1_pct",
                    "maximum_loading_pct",
                ]
            ],
            on=[
                "network",
                "scenario",
                "line",
            ],
            how="left",
        )

    else:

        line_combined = structural_combined.copy()

    # -----------------------------------------------------------------
    # Comparison
    # -----------------------------------------------------------------

    compare_networks(
        optimized_summary,
        iteration2_summary
    )

    # -----------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------

    header(
        "SAVING DIAGNOSTIC RESULTS"
    )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    summary_df = pd.DataFrame(
        [
            optimized_summary,
            iteration2_summary,
        ]
    )

    summary_df.to_csv(
        OUTPUT_SUMMARY,
        index=False
    )

    print(
        f"OK: Summary saved:\n"
        f"{OUTPUT_SUMMARY}"
    )

    bus_combined.to_csv(
        OUTPUT_BUSES,
        index=False
    )

    print(
        f"OK: Bus diagnostic saved:\n"
        f"{OUTPUT_BUSES}"
    )

    subnetwork_combined.to_csv(
        OUTPUT_SUBNETWORKS,
        index=False
    )

    print(
        f"OK: Subnetwork diagnostic saved:\n"
        f"{OUTPUT_SUBNETWORKS}"
    )

    line_combined.to_csv(
        OUTPUT_LINES,
        index=False
    )

    print(
        f"OK: Line diagnostic saved:\n"
        f"{OUTPUT_LINES}"
    )

    # -----------------------------------------------------------------
    # Final interpretation
    # -----------------------------------------------------------------

    final_interpretation(
        optimized_summary,
        iteration2_summary,
    )

    # -----------------------------------------------------------------
    # Completion
    # -----------------------------------------------------------------

    header(
        "S2 AC CONVERGENCE DIAGNOSTIC COMPLETE"
    )

    print()
    print(
        "OUTPUT FILES"
    )

    print(
        f"  {OUTPUT_SUMMARY}"
    )

    print(
        f"  {OUTPUT_SUBNETWORKS}"
    )

    print(
        f"  {OUTPUT_BUSES}"
    )

    print(
        f"  {OUTPUT_LINES}"
    )

    print()
    print(
        "NEXT STEP"
    )

    print(
        "Inspect s2_subnetwork_diagnostic.csv first."
    )

    print(
        "It shows exactly which PyPSA AC sub-network "
        "converged, the Newton-Raphson error, and "
        "the number of iterations."
    )

    print()
    print(
        "Only after primary_pf_converged=True should "
        "the corresponding AC line-loading values be "
        "used for engineering conclusions."
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()