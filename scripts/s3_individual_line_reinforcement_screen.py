import pypsa
import numpy as np
import pandas as pd
import copy

# ============================================================
# S3 — INDIVIDUAL-LINE REINFORCEMENT SCREEN
# CORRECTED FOR p_set-BASED NETWORK
# ============================================================

NETWORK_FILE = "data/processed/eirgrid_optimized_network.nc"
SNAPSHOT = "S2_PEAK_DEMAND"

CANDIDATE_LINES = [
    "merged_way/257889771-220+1",
    "way/343436171-220",
    "merged_way/1231251986-220+2",
    "merged_way/61295764-220+1",
    "merged_relation/4872159-220+1",
]

REINFORCEMENTS = [1.25, 1.50]

# Diagnostic voltage threshold
VOLTAGE_LIMIT = 0.90

# ============================================================
# HEADER
# ============================================================

print("=" * 110)
print("S3 INDIVIDUAL-LINE REINFORCEMENT SCREEN")
print("=" * 110)

print(f"\nNetwork : {NETWORK_FILE}")
print(f"Snapshot: {SNAPSHOT}")

# ============================================================
# LOAD NETWORK
# ============================================================

n_original = pypsa.Network(NETWORK_FILE)

print("\nNETWORK")
print("-" * 110)
print(f"Buses        : {len(n_original.buses)}")
print(f"Lines        : {len(n_original.lines)}")
print(f"Transformers : {len(n_original.transformers)}")
print(f"Generators   : {len(n_original.generators)}")
print(f"Loads        : {len(n_original.loads)}")
print(f"Links        : {len(n_original.links)}")

# ============================================================
# READ OPERATING POINT FROM p_set
# ============================================================

generator_dispatch = (
    n_original.generators_t.p_set.loc[SNAPSHOT]
)

load_dispatch = (
    n_original.loads_t.p_set.loc[SNAPSHOT]
)

generation = generator_dispatch.sum()
load = load_dispatch.sum()

mismatch = generation - load

print("\nORIGINAL POWER BALANCE")
print("-" * 110)
print(f"Generation : {generation:.6f} MW")
print(f"Load       : {load:.6f} MW")
print(f"Mismatch   : {mismatch:.6f} MW")

# ============================================================
# IDENTIFY WIND / NON-WIND GENERATORS
# ============================================================

wind_mask = (
    n_original.generators["carrier"]
    .astype(str)
    .str.lower()
    .str.contains("wind", na=False)
)

wind_generators = n_original.generators.index[wind_mask]
other_generators = n_original.generators.index[~wind_mask]

wind_generation = generator_dispatch.loc[
    wind_generators
].sum()

other_generation = generator_dispatch.loc[
    other_generators
].sum()

required_other_generation = (
    load - wind_generation
)

print("\nBALANCE CORRECTION")
print("-" * 110)
print(f"Wind generation           : {wind_generation:.6f} MW")
print(f"Other generation          : {other_generation:.6f} MW")
print(
    f"Required other generation : "
    f"{required_other_generation:.6f} MW"
)

# ============================================================
# CREATE BALANCED NETWORK
# ============================================================

n_base = copy.deepcopy(n_original)

current_other = (
    n_base.generators_t.p_set.loc[
        SNAPSHOT,
        other_generators
    ].sum()
)

if abs(current_other) < 1e-12:
    raise RuntimeError(
        "Non-wind generation is zero."
    )

balance_factor = (
    required_other_generation / current_other
)

n_base.generators_t.p_set.loc[
    SNAPSHOT,
    other_generators
] *= balance_factor

# ============================================================
# VERIFY BALANCE
# ============================================================

balanced_generation = (
    n_base.generators_t.p_set.loc[
        SNAPSHOT
    ].sum()
)

balanced_load = (
    n_base.loads_t.p_set.loc[
        SNAPSHOT
    ].sum()
)

balanced_mismatch = (
    balanced_generation - balanced_load
)

print(
    f"\nBalanced generation : "
    f"{balanced_generation:.12f} MW"
)

print(
    f"Balanced load       : "
    f"{balanced_load:.12f} MW"
)

print(
    f"Balanced mismatch   : "
    f"{balanced_mismatch:.12f} MW"
)

if abs(balanced_mismatch) > 1e-6:
    raise RuntimeError(
        "Balance correction failed."
    )

# ============================================================
# VERIFY CANDIDATE LINES
# ============================================================

print("\nCANDIDATE VERIFICATION")
print("-" * 110)

for line in CANDIDATE_LINES:

    if line not in n_base.lines.index:
        raise RuntimeError(
            f"Candidate line NOT FOUND: {line}"
        )

    s_nom = n_base.lines.at[
        line,
        "s_nom"
    ]

    print(
        f"{line:55s} "
        f"s_nom = {s_nom:.6f} MW"
    )

# ============================================================
# RESULTS
# ============================================================

results = []

# ============================================================
# TEST EACH LINE
# ============================================================

for candidate in CANDIDATE_LINES:

    original_s_nom = float(
        n_base.lines.at[
            candidate,
            "s_nom"
        ]
    )

    for multiplier in REINFORCEMENTS:

        print("\n")
        print("=" * 110)
        print(
            f"TEST — {candidate} — "
            f"{multiplier:.2f}x REINFORCEMENT"
        )
        print("=" * 110)

        # ----------------------------------------------------
        # FRESH COPY
        # ----------------------------------------------------

        n = copy.deepcopy(n_base)

        # ----------------------------------------------------
        # REINFORCE ONLY THIS LINE
        # ----------------------------------------------------

        new_s_nom = (
            original_s_nom * multiplier
        )

        n.lines.at[
            candidate,
            "s_nom"
        ] = new_s_nom

        print("\nREINFORCEMENT")
        print("-" * 110)
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

        # ----------------------------------------------------
        # AC POWER FLOW
        # ----------------------------------------------------

        converged = False
        iterations = np.nan
        final_error = np.nan

        try:

            print("\nRUNNING AC POWER FLOW — λ = 1.000")
            print("-" * 110)

            n.pf(
                snapshots=[SNAPSHOT],
                x_tol=1e-8,
                use_seed=True,
            )

            # ------------------------------------------------
            # CONVERGENCE
            # ------------------------------------------------

            try:

                pf_converged = (
                    n.sub_networks_t.pf_converged
                )

                if SNAPSHOT in pf_converged.index:

                    converged = bool(
                        np.all(
                            pf_converged.loc[
                                SNAPSHOT
                            ]
                        )
                    )

            except Exception:

                # Fallback based on finite voltage solution

                v = (
                    n.buses_t.v_mag_pu.loc[
                        SNAPSHOT
                    ]
                )

                converged = bool(
                    np.all(np.isfinite(v))
                    and np.all(v > 0)
                )

            # ------------------------------------------------
            # ITERATIONS
            # ------------------------------------------------

            try:

                iterations = int(
                    n.sub_networks_t.iteration.loc[
                        SNAPSHOT
                    ].max()
                )

            except Exception:

                iterations = np.nan

            # ------------------------------------------------
            # ERROR
            # ------------------------------------------------

            try:

                final_error = float(
                    n.sub_networks_t.error.loc[
                        SNAPSHOT
                    ].max()
                )

            except Exception:

                final_error = np.nan

        except Exception as exc:

            print("\nAC POWER FLOW EXCEPTION")
            print("-" * 110)
            print(
                type(exc).__name__,
                ":",
                exc
            )

            converged = False

        # ====================================================
        # FAILED CASE
        # ====================================================

        if not converged:

            print("\nRESULT")
            print("-" * 110)
            print("Converged       : FALSE")
            print("Voltage state   : INVALID / DIVERGED")

            results.append({

                "candidate_line": candidate,

                "reinforcement":
                    multiplier,

                "original_s_nom_mw":
                    original_s_nom,

                "new_s_nom_mw":
                    new_s_nom,

                "converged":
                    False,

                "iterations":
                    iterations,

                "error":
                    final_error,

                "min_v_pu":
                    np.nan,

                "min_v_bus":
                    None,

                "max_v_pu":
                    np.nan,

                "max_v_bus":
                    None,

                "max_line_loading_pct":
                    np.nan,

                "max_loaded_line":
                    None,

                "reinforced_line_loading_pct":
                    np.nan,

                "overloaded_lines":
                    np.nan,

                "max_transformer_loading_pct":
                    np.nan,

                "max_loaded_transformer":
                    None,

                "low_voltage_flag":
                    True,
            })

            continue

        # ====================================================
        # VOLTAGE ANALYSIS
        # ====================================================

        v_mag = (
            n.buses_t.v_mag_pu.loc[
                SNAPSHOT
            ]
        )

        min_v_bus = v_mag.idxmin()
        max_v_bus = v_mag.idxmax()

        min_v = float(
            v_mag.min()
        )

        max_v = float(
            v_mag.max()
        )

        # ====================================================
        # LINE LOADING
        # ====================================================

        line_loading = (
            n.lines_t.p0.loc[
                SNAPSHOT
            ].abs()
            /
            n.lines.s_nom
            *
            100.0
        )

        max_loaded_line = (
            line_loading.idxmax()
        )

        max_line_loading = float(
            line_loading.max()
        )

        reinforced_line_loading = float(
            line_loading.loc[
                candidate
            ]
        )

        overloaded_lines = int(
            (
                line_loading > 100.0
            ).sum()
        )

        # ====================================================
        # TRANSFORMER LOADING
        # ====================================================

        if len(n.transformers) > 0:

            transformer_loading = (
                n.transformers_t.p0.loc[
                    SNAPSHOT
                ].abs()
                /
                n.transformers.s_nom
                *
                100.0
            )

            max_transformer = (
                transformer_loading.idxmax()
            )

            max_transformer_loading = float(
                transformer_loading.max()
            )

        else:

            max_transformer = None
            max_transformer_loading = np.nan

        # ====================================================
        # PRINT RESULT
        # ====================================================

        print("\nRESULT")
        print("-" * 110)

        print(
            "Converged                  : TRUE"
        )

        print(
            f"Iterations                 : "
            f"{iterations}"
        )

        print(
            f"Final error                : "
            f"{final_error:.6e}"
        )

        print(
            f"Min V magnitude            : "
            f"{min_v:.6f} pu"
        )

        print(
            f"Min-V bus                  : "
            f"{min_v_bus}"
        )

        print(
            f"Max V magnitude            : "
            f"{max_v:.6f} pu"
        )

        print(
            f"Max-V bus                  : "
            f"{max_v_bus}"
        )

        print(
            f"Max line loading           : "
            f"{max_line_loading:.6f} %"
        )

        print(
            f"Max-loaded line            : "
            f"{max_loaded_line}"
        )

        print(
            f"Reinforced-line loading    : "
            f"{reinforced_line_loading:.6f} %"
        )

        print(
            f"Overloaded lines (>100%)   : "
            f"{overloaded_lines}"
        )

        print(
            f"Max transformer loading    : "
            f"{max_transformer_loading:.6f} %"
        )

        print(
            f"Max-loaded transformer     : "
            f"{max_transformer}"
        )

        # ====================================================
        # STORE
        # ====================================================

        results.append({

            "candidate_line":
                candidate,

            "reinforcement":
                multiplier,

            "original_s_nom_mw":
                original_s_nom,

            "new_s_nom_mw":
                new_s_nom,

            "converged":
                True,

            "iterations":
                iterations,

            "error":
                final_error,

            "min_v_pu":
                min_v,

            "min_v_bus":
                min_v_bus,

            "max_v_pu":
                max_v,

            "max_v_bus":
                max_v_bus,

            "max_line_loading_pct":
                max_line_loading,

            "max_loaded_line":
                max_loaded_line,

            "reinforced_line_loading_pct":
                reinforced_line_loading,

            "overloaded_lines":
                overloaded_lines,

            "max_transformer_loading_pct":
                max_transformer_loading,

            "max_loaded_transformer":
                max_transformer,

            "low_voltage_flag":
                min_v < VOLTAGE_LIMIT,
        })

# ============================================================
# FINAL SUMMARY
# ============================================================

results_df = pd.DataFrame(
    results
)

print("\n")
print("=" * 110)
print("S3 FINAL REINFORCEMENT SCREEN SUMMARY")
print("=" * 110)

summary_columns = [

    "candidate_line",

    "reinforcement",

    "converged",

    "min_v_pu",

    "max_line_loading_pct",

    "reinforced_line_loading_pct",

    "overloaded_lines",

    "max_transformer_loading_pct",

    "low_voltage_flag",
]

print(
    results_df[
        summary_columns
    ].to_string(
        index=False
    )
)

# ============================================================
# RANKING
# ============================================================

valid = results_df[
    results_df["converged"] == True
].copy()

if len(valid) > 0:

    # Primary objective:
    # lower maximum loading
    #
    # Secondary:
    # fewer overloaded lines
    #
    # Voltage below 0.90 pu receives penalty.

    valid["score"] = (

        valid[
            "max_line_loading_pct"
        ]

        +

        10.0
        *
        valid[
            "overloaded_lines"
        ]

        +

        100.0
        *
        valid[
            "low_voltage_flag"
        ].astype(int)
    )

    ranked = valid.sort_values(
        "score"
    )

    print("\n")
    print("=" * 110)
    print(
        "S3 RANKING — LOWER SCORE IS BETTER"
    )
    print("=" * 110)

    print(
        ranked[
            [
                "candidate_line",
                "reinforcement",
                "min_v_pu",
                "max_line_loading_pct",
                "reinforced_line_loading_pct",
                "overloaded_lines",
                "score",
            ]
        ].to_string(
            index=False
        )
    )

# ============================================================
# SAVE DIAGNOSTIC RESULTS ONLY
# ============================================================

OUTPUT_FILE = (
    "data/processed/"
    "s3_individual_line_reinforcement_results.csv"
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ============================================================
# END
# ============================================================

print("\n")
print("=" * 110)
print("S3 COMPLETE")
print("=" * 110)

print(
    f"\nDiagnostic results saved to:"
)

print(
    OUTPUT_FILE
)

print("\nIMPORTANT:")
print(
    "Original network was NOT modified."
)

print(
    "Original network was NOT overwritten."
)

print(
    "No reinforced network was saved."
)

print(
    "Only the S3 diagnostic CSV was saved."
)