import pypsa

INPUT = "data/processed/ireland_network.nc"

SOURCE = "way/516651650-220"
SINK = "way/1003262502-220"

n = pypsa.Network(INPUT)

# ------------------------------------------------------------
# Temporary operating condition
# ------------------------------------------------------------

# Slack/reference generator
n.add(
    "Generator",
    "TEST_SLACK",
    bus=SOURCE,
    control="Slack",
    p_set=500,
)

# Temporary load
n.add(
    "Load",
    "TEST_LOAD",
    bus=SINK,
    p_set=500,
)

print("=== TEMPORARY POWER-FLOW TEST ===")
print("Generator:", SOURCE)
print("Load:", SINK)
print("Power transfer: 500 MW")

# ------------------------------------------------------------
# Solve linear power flow
# ------------------------------------------------------------

n.lpf()

print()
print("=== POWER FLOW RESULT ===")

print("Generator output:")
print(n.generators_t.p.to_string())

print()
print("Bus voltage angles:")
print(n.buses_t.v_ang.to_string())

print()
print("Line loading:")

line_loading = (
    n.lines_t.p0.abs()
    / n.lines.s_nom
    * 100
)

print(line_loading.T.to_string())

print()
print("Maximum line loading: %.2f%%" % line_loading.max().max())

print()
print("Power-flow test completed.")