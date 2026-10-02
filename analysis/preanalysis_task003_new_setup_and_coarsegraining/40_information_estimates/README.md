# 40_information_estimates — path divergence, mutual information, efficiencies

Computed on the 4-state epistemic chain with `../scripts/estimate.py`. Because
the state space is small and the process is approximately Markov, path KL is
**exact** given the fitted kernels:

```
D(P‖Q) over paths = D(p₀‖q₀) + Σ_t E_P[ D( P(·|s_t) ‖ Q(·|s_t) ) ]
```

No variational bound, no fitted critic. That is the point of the coarse-graining:
the earlier critic-based estimates on the 325-state vote vector failed every
reliability check.

| file | contents |
|---|---|
| `estimates_chain.csv` | per ρ at horizon 14, with bias-corrected cluster-bootstrap intervals |
| `estimates_curves.csv` | the same as a function of horizon h = 1..14 |

## Results at h = 14

| ρ | K(A2‖silent) nats | I(intervention; state) bits | I(target; state) bits | EP silent | EP A2 | posts/episode |
|---|---:|---:|---:|---:|---:|---:|
| 0.75 | **3.045** [2.77, 3.28] | **0.294** [0.26, 0.33] | **0.022** | 0.015 | 0.069 | 69.1 |
| 1.00 | **1.931** [1.60, 2.31] | **0.154** [0.12, 0.19] | **0.004** | 0.390 | 0.045 | 83.9 |

## How to read these

**K(A2‖silent)** is how far the decoy-targeting controller displaces the
trajectory law from silence. It is **larger at ρ = 0.75 than at ρ = 1** (3.05 vs
1.93) despite *fewer* posts — consistent with the refresh mechanism: at finite
persistence the controller's facts keep being re-injected into decaying memories.

**I(intervention; state)** — at most 1 bit — is how recognisable intervention is
from the population state alone. 0.29 bits at ρ = 0.75.

**I(target; state) ≈ 0.022 bits** is the key number: which target the controller
was given is almost invisible in the trajectory. This is the quantitative form of
the pool problem — both arms quote the same facts.

**Efficiency** `η = I(target;·)/C_π` is therefore **0.006–0.018**, essentially
zero, and is *confounded*: it measures the design as executed, not truth-directed
steering.

## Two warnings

**`EP_*` is not physical dissipation.** It is the Markov-chain irreversibility
`Σ_ij p_i T_ij ln(p_i T_ij / p_j T_ji)`. No reverse protocol has been justified,
and the reference document is explicit that a sequence-reversal score must not be
relabelled entropy production. Note it also *decreases with horizon* — these are
transient chains, not stationary ones.

**Validity differs by cell.** Only ρ = 1 / silent passes all four Markov tests.
At ρ = 0.75 these numbers carry ~0.015 nats/step of unexplained memory and a
Chapman–Kolmogorov gap reaching 0.11 at five steps. Treat long-horizon path
quantities there as indicative.

## Bootstrap note

Resampling episodes with replacement smooths the refitted kernel and biases KL
downward, so raw percentile intervals excluded the point estimates. The intervals
above are **bias-corrected** (shifted by bootstrap mean minus point estimate).
