from pathlib import Path
import numpy as np
import pandas as pd
import pypsa


# ============================================================
# IRELAND GRID - S2 VOLTAGE SUPPORT SENSITIVITY
# ============================================================

NETWORK_PATH = Path(
    "data/processed/eirgrid_second_reinforced_network.nc"
)

SNAPSHOT = "S2_PEAK_DEMAND"

OUTPUT_PATH = Path(
    "data/processed/s2_voltage_support_sensitivity.csv"
)

# Weak-voltage bus identified from the validated S2 screen
WEAK_BUS = "way/104388595-220"

# Reactive support values to test
Q_SUPPORT_VALUES = [
    0,
    25,
    50,
    75,
    100,
    125,
    150,
    200,
    250,
    300,
    400,
    500,
]


# ============================================================
# HELPERS
# ============================================================

def load_network():
    """
    Load a fresh copy of the source network for every test.

    Source network is never modified on disk.
    """
    return pypsa.Network(NETWORK_PATH)


def run_pf(network):
    """
    Run AC nonlinear PF using the same basic solver setup
    as the validated S2 targeted reinforcement screen.

    IMPORTANT:
    Do NOT use use_seed=True here.
    """

    result = network.pf(
        snapshots=[SNAPSHOT],
        x_tol=1e-6,
    )

    return result


def get_pf_convergence(network, pf_result):
    """
    Determine convergence conservatively.

    We do NOT trust network.pf_converged alone because
    the previous sensitivity script reported True despite
    PyPSA explicitly warning that PF did not converge.
    """

    # First inspect the returned PF result when available.
    try:
        if isinstance(pf_result, dict):

            if "converged" in pf_result:

                value = pf_result["converged"]

                if hasattr(value, "loc"):
                    value = value.loc[SNAPSHOT]

                return bool(value)

    except Exception:
        pass

    # Fallback to network.pf_converged.
    try:

        value = network.pf_converged.loc[SNAPSHOT]

        return bool(value)

    except Exception:

        return False


def get_pf_metrics(network, pf_result):
    """
    Extract physically meaningful PF metrics.

    A result is considered invalid if:
      - PF did not converge
      - voltage values are non-finite
      - voltage values are physically absurd
      - line loading is non-finite
    """

    converged = get_pf_convergence(
        network,
        pf_result
    )

    # --------------------------------------------------------
    # Voltage
    # --------------------------------------------------------

    try:

        v_mag = network.buses_t.v_mag_pu.loc[SNAPSHOT]

        finite_v = (
            v_mag
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

    except Exception:

        finite_v = pd.Series(dtype=float)

    if len(finite_v):

        min_voltage = float(
            finite_v.min()
        )

        min_voltage_bus = str(
            finite_v.idxmin()
        )

        # Reject clearly corrupted PF results.
        # Transmission-network voltages should not be
        # anywhere remotely close to 1e18 pu.
        if (
            not np.isfinite(min_voltage)
            or min_voltage <= 0
            or min_voltage > 2.0
        ):

            converged = False

    else:

        min_voltage = np.nan
        min_voltage_bus = ""
        converged = False


    # --------------------------------------------------------
    # AC line loading
    # --------------------------------------------------------

    if len(network.lines):

        s_nom = (
            network.lines["s_nom"]
            .replace(0, np.nan)
        )

        try:

            p0 = network.lines_t.p0.loc[
                SNAPSHOT
            ]

            q0 = network.lines_t.q0.loc[
                SNAPSHOT
            ]

            apparent_power = np.sqrt(
                p0.pow(2) + q0.pow(2)
            )

            loading = (
                100.0
                * apparent_power
                / s_nom
            )

            finite_loading = (
                loading
                .replace(
                    [np.inf, -np.inf],
                    np.nan
                )
                .dropna()
            )

        except Exception:

            finite_loading = pd.Series(
                dtype=float
            )

    else:

        finite_loading = pd.Series(
            dtype=float
        )


    if len(finite_loading):

        max_loading = float(
            finite_loading.max()
        )

        worst_line = str(
            finite_loading.idxmax()
        )

        lines_over_100 = int(
            (finite_loading > 100).sum()
        )

        lines_over_110 = int(
            (finite_loading > 110).sum()
        )

        lines_over_120 = int(
            (finite_loading > 120).sum()
        )

        # Reject corrupted numerical results.
        if (
            not np.isfinite(max_loading)
            or max_loading < 0
            or max_loading > 100000
        ):

            converged = False

    else:

        max_loading = np.nan
        worst_line = ""
        lines_over_100 = 0
        lines_over_110 = 0
        lines_over_120 = 0

        converged = False


    return {
        "pf_converged": bool(converged),
        "min_voltage_pu": min_voltage,
        "min_voltage_bus": min_voltage_bus,
        "max_line_loading_pct": max_loading,
        "worst_line": worst_line,
        "lines_over_100": lines_over_100,
        "lines_over_110": lines_over_110,
        "lines_over_120": lines_over_120,
    }


def add_reactive_support(network, q_mvar):
    """
    Add temporary local reactive support at the weak bus.

    This is an in-memory experiment only.

    P = 0 MW
    Q = +q_mvar MVAr

    Positive Q represents reactive injection into the network.
    """

    name = "TEMP_S2_VOLTAGE_SUPPORT"

    if name in network.generators.index:

        network.generators.drop(
            index=name,
            inplace=True
        )

    network.add(
        "Generator",
        name,
        bus=WEAK_BUS,

        # Give the temporary device a non-zero
        # nominal rating so the PF model has a valid
        # generator object.
        p_nom=max(
            1.0,
            float(q_mvar)
        ),

        p_set=0.0,
        q_set=float(q_mvar),

        control="PQ",

        carrier="temporary_voltage_support",
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 110)
print("IRELAND GRID - S2 VOLTAGE SUPPORT SENSITIVITY")
print("=" * 110)

print()
print(f"Network       : {NETWORK_PATH}")
print(f"Snapshot      : {SNAPSHOT}")
print("PF            : AC nonlinear")
print("Slack         : distributed")
print("Dispatch      : unchanged")
print("Loads         : unchanged")
print("Source        : READ-ONLY")
print(f"Weak bus      : {WEAK_BUS}")
print()
print("Reactive support is TEMPORARY and IN-MEMORY only.")
print("=" * 110)


# ============================================================
# BASELINE
# ============================================================

print()
print("-" * 110)
print("BASELINE")
print("-" * 110)

baseline_network = load_network()

try:

    baseline_pf_result = run_pf(
        baseline_network
    )

    baseline = get_pf_metrics(
        baseline_network,
        baseline_pf_result
    )

except Exception as exc:

    print()
    print("BASELINE POWER FLOW ERROR")
    print(str(exc))

    raise RuntimeError(
        "Baseline PF failed. "
        "Do not continue."
    )


for key, value in baseline.items():

    print(
        f"{key:25s}: {value}"
    )


# ============================================================
# HARD BASELINE VALIDATION
# ============================================================

if not baseline["pf_converged"]:

    raise RuntimeError(
        "\nBASELINE POWER FLOW IS NOT VALID.\n"
        "The sensitivity study has been stopped.\n"
        "The source network must reproduce the validated "
        "S2 PF before voltage-support testing."
    )


print()
print("Baseline validated.")

print(
    f"Minimum voltage : "
    f"{baseline['min_voltage_pu']:.6f} pu"
)

print(
    f"Maximum loading : "
    f"{baseline['max_line_loading_pct']:.6f}%"
)


# ============================================================
# SENSITIVITY
# ============================================================

results = []

print()
print("=" * 110)
print("REACTIVE POWER SUPPORT SENSITIVITY")
print("=" * 110)


for q_mvar in Q_SUPPORT_VALUES:

    print()
    print("-" * 110)
    print(
        f"TEST: {q_mvar} MVAr REACTIVE SUPPORT"
    )
    print("-" * 110)

    network = load_network()

    add_reactive_support(
        network,
        q_mvar
    )

    try:

        pf_result = run_pf(
            network
        )

        metrics = get_pf_metrics(
            network,
            pf_result
        )

    except Exception as exc:

        print()
        print("POWER FLOW ERROR")
        print(str(exc))

        metrics = {
            "pf_converged": False,
            "min_voltage_pu": np.nan,
            "min_voltage_bus": "",
            "max_line_loading_pct": np.nan,
            "worst_line": "",
            "lines_over_100": np.nan,
            "lines_over_110": np.nan,
            "lines_over_120": np.nan,
        }


    if (
        metrics["pf_converged"]
        and np.isfinite(
            metrics["min_voltage_pu"]
        )
    ):

        voltage_change = (
            metrics["min_voltage_pu"]
            - baseline["min_voltage_pu"]
        )

    else:

        voltage_change = np.nan


    if (
        metrics["pf_converged"]
        and np.isfinite(
            metrics["max_line_loading_pct"]
        )
    ):

        loading_change = (
            metrics["max_line_loading_pct"]
            - baseline["max_line_loading_pct"]
        )

    else:

        loading_change = np.nan


    row = {

        "snapshot": SNAPSHOT,

        "q_support_mvar": q_mvar,

        "pf_converged":
            metrics["pf_converged"],

        "min_voltage_pu":
            metrics["min_voltage_pu"],

        "min_voltage_bus":
            metrics["min_voltage_bus"],

        "voltage_change_pu":
            voltage_change,

        "max_line_loading_pct":
            metrics["max_line_loading_pct"],

        "loading_change_pct_points":
            loading_change,

        "worst_line":
            metrics["worst_line"],

        "lines_over_100":
            metrics["lines_over_100"],

        "lines_over_110":
            metrics["lines_over_110"],

        "lines_over_120":
            metrics["lines_over_120"],
    }


    results.append(row)


    print(
        f"PF converged       : "
        f"{metrics['pf_converged']}"
    )

    print(
        f"Minimum voltage    : "
        f"{metrics['min_voltage_pu']}"
    )

    print(
        f"Voltage change     : "
        f"{voltage_change}"
    )

    print(
        f"Maximum loading    : "
        f"{metrics['max_line_loading_pct']}"
    )

    print(
        f"Loading change     : "
        f"{loading_change}"
    )

    print(
        f"Lines >100%        : "
        f"{metrics['lines_over_100']}"
    )


# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(
    results
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 110)
print("S2 VOLTAGE SUPPORT SENSITIVITY SUMMARY")
print("=" * 110)

print(
    results_df[
        [
            "q_support_mvar",
            "pf_converged",
            "min_voltage_pu",
            "voltage_change_pu",
            "max_line_loading_pct",
            "loading_change_pct_points",
            "lines_over_100",
        ]
    ].to_string(index=False)
)


# ============================================================
# BEST VALIDATED VOLTAGE RESULT
# ============================================================

valid = results_df[
    (results_df["pf_converged"] == True)
    & (
        results_df["min_voltage_pu"]
        .notna()
    )
].copy()


if len(valid):

    best_voltage = valid.loc[
        valid["min_voltage_pu"].idxmax()
    ]

    print()
    print("=" * 110)
    print("BEST VALIDATED VOLTAGE RESULT TESTED")
    print("=" * 110)

    print(
        f"Reactive support : "
        f"{best_voltage['q_support_mvar']:.0f} MVAr"
    )

    print(
        f"Minimum voltage  : "
        f"{best_voltage['min_voltage_pu']:.6f} pu"
    )

    print(
        f"Voltage change   : "
        f"{best_voltage['voltage_change_pu']:+.6f} pu"
    )

    print(
        f"Maximum loading  : "
        f"{best_voltage['max_line_loading_pct']:.6f}%"
    )

    print(
        f"Lines >100%      : "
        f"{int(best_voltage['lines_over_100'])}"
    )


else:

    print()
    print("=" * 110)
    print("NO VALIDATED REACTIVE SUPPORT RESULT")
    print("=" * 110)


# ============================================================
# AUDIT
# ============================================================

print()
print("=" * 110)
print("AUDIT FILE SAVED")
print("=" * 110)

print(OUTPUT_PATH)

print()
print("NO SOURCE NETWORK MODIFICATION PERFORMED.")
print("NO DISPATCH MODIFICATION PERFORMED.")
print("NO LOAD MODIFICATION PERFORMED.")
print("ALL VOLTAGE SUPPORT WAS TEMPORARY IN-MEMORY TESTING.")

print()
print("=" * 110)
print("S2 VOLTAGE SUPPORT SENSITIVITY COMPLETE")
print("=" * 110)