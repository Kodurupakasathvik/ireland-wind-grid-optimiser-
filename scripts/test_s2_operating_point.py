import pypsa
import numpy as np

INPUT = r"data\processed\eirgrid_optimized_network.nc"

print("=" * 70)
print("S2 PEAK DEMAND - OPERATING POINT TEST")
print("=" * 70)

n = pypsa.Network(INPUT)

snap = "S2_PEAK_DEMAND"

# ------------------------------------------------------------
# 1. COPY NETWORK IN MEMORY
# ------------------------------------------------------------

print("\nCreating in-memory diagnostic copy...")
print("Original network will NOT be modified.")

# ------------------------------------------------------------
# 2. CHECK REQUIRED COMPONENTS
# ------------------------------------------------------------

required_generators = [
    "ewic_import",
    "greenlink_import",
]

required_links = [
    "EWIC_interface",
    "Greenlink_interface",
]

for name in required_generators:
    if name not in n.generators.index:
        raise RuntimeError(f"Missing generator: {name}")

for name in required_links:
    if name not in n.links.index:
        raise RuntimeError(f"Missing link: {name}")

# ------------------------------------------------------------
# 3. READ IMPORT VALUES
# ------------------------------------------------------------

ewic_p = float(n.generators_t.p_set.at[snap, "ewic_import"])
greenlink_p = float(
    n.generators_t.p_set.at[snap, "greenlink_import"]
)

print("\nINTERCONNECTOR IMPORTS")
print("-" * 70)

print(f"EWIC      : {ewic_p:.6f} MW")
print(f"Greenlink : {greenlink_p:.6f} MW")
print(f"Total     : {ewic_p + greenlink_p:.6f} MW")

# ------------------------------------------------------------
# 4. WRITE INTERCONNECTOR LINK DISPATCH
# ------------------------------------------------------------

print("\nSETTING S2 INTERCONNECTOR LINK FLOWS")
print("-" * 70)

for link_name, value in [
    ("EWIC_interface", ewic_p),
    ("Greenlink_interface", greenlink_p),
]:

    if link_name not in n.links_t.p_set.columns:
        n.links_t.p_set[link_name] = 0.0

    n.links_t.p_set.at[snap, link_name] = value

    print(f"{link_name}: S2 p_set = {value:.6f} MW")

# ------------------------------------------------------------
# 5. CHECK ACTIVE POWER BALANCE
# ------------------------------------------------------------

generation = float(
    n.generators_t.p_set.loc[snap].sum()
)

load = float(
    n.loads_t.p_set.loc[snap].sum()
)

print("\nACTIVE POWER BALANCE")
print("-" * 70)

print(f"Generation : {generation:.6f} MW")
print(f"Load       : {load:.6f} MW")
print(f"Difference : {generation - load:.6f} MW")

# ------------------------------------------------------------
# 6. IDENTIFY AC GENERATORS
# ------------------------------------------------------------

print("\nGENERATOR CONTROLS")
print("-" * 70)

print(
    n.generators[
        ["bus", "carrier", "control", "p_nom"]
    ].to_string()
)

# ------------------------------------------------------------
# 7. TEMPORARILY CREATE A SLACK REFERENCE
# ------------------------------------------------------------

print("\nSETTING TEMPORARY SLACK REFERENCE")
print("-" * 70)

# Use the main Irish generation bus as the diagnostic slack.
slack_generator = "eirgrid_non_wind_generation"

if slack_generator not in n.generators.index:
    raise RuntimeError(
        f"Slack generator {slack_generator} not found"
    )

original_controls = n.generators.control.copy()

n.generators.loc[slack_generator, "control"] = "Slack"

print(
    f"{slack_generator} -> control = "
    f"{n.generators.loc[slack_generator, 'control']}"
)

# ------------------------------------------------------------
# 8. REACTIVE POWER CHECK
# ------------------------------------------------------------

print("\nREACTIVE POWER DATA")
print("-" * 70)

print("Generator q_set columns:",
      list(n.generators_t.q_set.columns))

print("Load q_set columns:",
      list(n.loads_t.q_set.columns))

# ------------------------------------------------------------
# 9. RUN AC POWER FLOW
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("RUNNING S2 AC POWER FLOW")
print("=" * 70)

try:

    n.pf(
        snapshots=[snap],
        x_tol=1e-6,
        use_seed=True,
        distribute_slack=False,
    )

    print("\nAC POWER FLOW RETURNED.")

except Exception as exc:

    print("\nAC POWER FLOW RAISED AN EXCEPTION:")
    print(type(exc).__name__, str(exc))

# ------------------------------------------------------------
# 10. CONVERGENCE CHECK
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("AC RESULT")
print("=" * 70)

try:

    print(
        "Converged:",
        n.sub_networks_t.pf_converged.loc[snap].to_dict()
    )

except Exception:

    print(
        "Could not read pf_converged directly."
    )

# ------------------------------------------------------------
# 11. BUS VOLTAGE CHECK
# ------------------------------------------------------------

print("\nBUS VOLTAGES")

if hasattr(n, "buses_t") and hasattr(n.buses_t, "v_mag_pu"):

    values = n.buses_t.v_mag_pu.loc[snap]

    finite = values[np.isfinite(values)]

    if len(finite) > 0:

        print(
            f"Minimum voltage : {finite.min():.6f} pu"
        )

        print(
            f"Maximum voltage : {finite.max():.6f} pu"
        )

    else:

        print("No finite AC voltage results.")

# ------------------------------------------------------------
# 12. LINE LOADING ONLY IF PF PRODUCED VALID RESULTS
# ------------------------------------------------------------

print("\nLINE RESULTS")

try:

    if hasattr(n.lines_t, "p0"):

        p0 = n.lines_t.p0.loc[snap]

        finite = p0[np.isfinite(p0)]

        if len(finite) > 0:

            print(
                "Finite line-flow results:",
                len(finite)
            )

            print(
                "Maximum absolute line flow:",
                float(np.max(np.abs(finite))),
                "MW"
            )

        else:

            print("No finite line-flow results.")

except Exception as exc:

    print(
        "Could not inspect line results:",
        exc
    )

print("\n" + "=" * 70)
print("OPERATING POINT TEST COMPLETE")
print("=" * 70)

print("\nIMPORTANT:")
print("This script does NOT save or modify the network.")
print("No line reinforcement was performed.")