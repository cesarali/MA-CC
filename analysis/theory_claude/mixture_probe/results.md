# Mixture probe — results (28 Oct 2026)

Pre-registration: `00_preregistration_copy.md` (= blackboard `2026-10-28_claude_09.md`, committed before computing).
Normal continuation, controlled arm, q 6, 300 episodes per cell. Files: `probe_cells.csv`, `probe_summary.csv`.

**Population susceptibility to the encountered controller mixture**, S = E_M E_{f∼w}[P(A2 | M + f) − P(A2 | M)],
from t = 30 to t = 130:

| setup, ρ | S(30) → S(130) | relative change | deletion probe (redundancy removed) | retained δ(130)/δ(30) in the simulation |
|---|---|---|---|---|
| OP no-proof, 0.25 | 0.029 → 0.019 | −36% | −33% | 0.37 |
| OP no-proof, 0.75 | 0.011 → 0.002 | −80% | −82% | 0.33 |
| OP symmetric, 0.25 | 0.043 → 0.040 | −8% | −3% | 0.67 |
| OP symmetric, 0.75 | 0.016 → 0.015 | −5% | +24% | (no effect to keep; δ ≈ 0) |
| task003 no-proof, 0.25 | 0.094 → 0.090 | −4% | −2% | 0.97 |
| task003 no-proof, 0.75 | 0.048 → 0.051 | +5% | +11% | 1.05 |
| task003 symmetric, 0.25 | 0.084 → 0.079 | −5% | +1% | 0.97 |
| task003 symmetric, 0.75 | 0.040 → 0.040 | −1% | +4% | 0.75 |

- **S tracks persistence:** a large drop where OP no-proof fades, and roughly flat where task003 persists. OP
  symmetric ρ 0.25 is under-tracked (−8% against −33%).
- **Not redundancy:** the share of encountered facts already held rises by 7–10 points, but the deletion probe shows
  the same drops.

**Decomposition** (pre-registered prediction: OP mixture term negative and larger than the context term; task003
mixture term ≈ 0):

| setup, ρ | ΔS | mixture term (order 1) | context term (order 1) | mixture (order 2) | context (order 2) |
|---|---|---|---|---|---|
| OP no-proof, 0.25 | −0.0106 | −0.0111 ± 0.0014 | +0.0005 | −0.0026 | −0.0081 |
| OP no-proof, 0.75 | −0.0091 | −0.0093 ± 0.0009 | +0.0003 | −0.0055 | −0.0035 |
| task003 symmetric, 0.25 | −0.0046 | −0.0097 | +0.0052 | +0.0012 | −0.0057 |
| task003 no-proof, 0.25 | −0.0037 | −0.0087 | +0.0050 | −0.0010 | −0.0026 |

- **OP:** the mixture term is ≥ 25% of |ΔS| in both orders, so the null criterion is not met and a changing strength
  mixture is **not ruled out**. But the attribution depends on the order: about 100% in order 1, 25% and 60% in order
  2. Mixture and context interact, and the split is not identified.
- **task003: prediction not met.** The mixture term is also negative in order 1 (about −0.009), offset by context
  (+0.005). The encountered mixture drifts in both tasks; what differs is whether S as a whole falls.

**Reading for the hierarchy.** One population variable, the susceptibility S(t) to the encountered controller mixture,
separates fading from persistence. Whether its change belongs to level 2 (mixture) or level 3 (context) cannot be
decided by this decomposition. S itself is a candidate for the "one coarse context variable".
