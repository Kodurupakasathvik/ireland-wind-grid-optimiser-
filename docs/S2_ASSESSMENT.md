# S2 peak-demand AC assessment

## Release classification

**S2 is an unresolved full-reactive-power AC stress case, not evidence of a
confirmed physical collapse.** It is excluded from all AC-secure curtailment,
economic, and emissions claims.

The safe label for S2 in a portfolio or report is:

> A conditional AC stress-test diagnostic whose realistic full-Q operating
> point was not validated; controlled variants show severe voltage and thermal
> insecurity before the full-Q case becomes numerically invalid.

## Evidence retained in the repository

| Evidence | Finding | Consequence |
| --- | --- | --- |
| `s2_scenario_data_diagnostic.csv` | The early stored S2 snapshots have no finite non-zero generator or load `p_set` values. | They cannot establish a realistic AC operating point by themselves. |
| `s2_convergence_diagnostic.csv` | Standard and seeded Newton-Raphson solves fail. A distributed-slack calculation converges numerically but reaches 165–177% maximum line loading. | Numerical convergence under a modified slack treatment is not physical security. |
| `s4_5b_ac_convergence_diagnostic.csv` | Controlled reconstruction has 5,775.591 MW generation, 5,899.270 MW load, and a -123.679 MW pre-solve imbalance. | The reconstructed operating point needs explicit balancing assumptions. |
| `s4_5j_q_continuation.csv` | Controlled solves at 0–7.1% of the assumed reactive load converge, but minimum voltage is 0.641–0.739 pu and maximum line loading is at least 174%. At 7.2% and above, the numerical validation fails. | Even the conditional converged cases are not secure; the full-Q case is not validated. |

## What was ruled out and what was not

The diagnostics did not identify a simple bad branch parameter or a confirmed
network-islanding explanation. They instead point to an interaction between an
under-specified historical snapshot, active-power balancing choice, reactive
loading, and Newton-Raphson solvability. That is sufficient to reject S2 from
security claims, but insufficient to diagnose a real-system voltage collapse.

## Release treatment

- Keep S2 in figures and tables only as a labelled stress-test diagnostic.
- Do not report failed-S2 flow magnitudes as physical bottlenecks.
- Do not count S2 in AC-secure benefits or investment recommendations.
- Do not equate `non-converged` with `insecure`: non-convergence means the
  configured model did not yield a validated AC solution.

## If the study is extended

Resolve the source dispatch and reactive-power provenance before any further
S2 optimisation: confirm generator controls and Q limits, load power factors,
interconnector set-points, slack participation, and the exact timestamped
network topology. Re-run a balanced AC base case before testing any
reinforcement or value claim.
