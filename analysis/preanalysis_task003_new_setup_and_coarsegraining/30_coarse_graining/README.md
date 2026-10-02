# 02_markov_tests — is the coarse-graining valid?

The population state is coarse-grained to the **mean agent posterior**
`e = mean_j P(ALLOCATION_0 | K_j)`, binned into 4 states at cuts 0.33 / 0.50 / 0.75.
These files test whether the resulting chain is Markovian, which is the
precondition for computing path KL, entropy production and mutual information on it.

**Read [`four_tests_report_2026-10-02.md`](four_tests_report_2026-10-02.md) first**, then [`state_space_decision_2026-10-02.md`](state_space_decision_2026-10-02.md).
It supersedes the earlier single-test report.

| file | what it is |
|---|---|
| `four_tests_report_2026-10-02.md` | **main report** — 4 independent tests, which coordinate is missing, what to use |
| `coarse_graining_recipe_2026-10-02.md` | how to build the coarse-graining; what worked and what did not |
| `markov_test_report_2026-10-02.md` | the first attempt; its cross-arm transfer test is invalid (explained inside) and its conclusion was too generous |
| `four_tests.csv` | order / Chapman-Kolmogorov / conditional-independence results per (ρ, arm) |
| `lumpability.csv` | TV gap within each macrostate, per candidate hidden coordinate |
| `augmented_states.csv` | order gain after adding each coordinate as a second dimension |
| `bin_sweep.csv` | 2-6 bins, quantile and semantic schemes |
| `markov_results*.json`, `markov_order.json` | raw outputs of the first attempt |

## The result in one line

The chain is Markovian **only in the silent arm at ρ = 1.00**; at ρ = 0.75 it is
an approximation carrying ~0.015 nats/step of unexplained memory, and the missing
coordinate is knowledge **spread** under silence and **survival of the decisive
facts** under control.
