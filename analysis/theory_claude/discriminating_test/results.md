# Discriminating test — results (28 Oct 2026)

Pre-registration: `00_preregistration.md` (commit before computing). Data: silent-arm memories at t = 30, q 3/6,
ρ 0.25/0.5/0.75, each paired with every controller-exclusive fact; train on q = 6 even episodes, evaluate on q = 3 odd
episodes. 315k–432k held-out pairs per setup.

**Pre-registered result** (`results_by_setup.csv`):

| setup | R², count response (A) | R², count + combinations (B) | gain | within-count share | combination needed? |
|---|---|---|---|---|---|
| task003 symmetric | −0.022 | 0.009 | 0.032 | −0.005 | no |
| task003 no-proof | −0.008 | −0.003 | 0.004 | −0.007 | no |
| OP symmetric | −0.010 | −0.014 | −0.004 | −0.011 | no |
| OP no-proof | −0.003 | −0.002 | 0.001 | 0.000 | no |

- By the rule, a combination term is not needed anywhere. But the count response **also fails**: R² ≈ 0 in all four
  setups.
- **Expectation not met.** I had expected count to suffice in task003 and combinations to be needed in OP. Neither
  summary of the agent's own memory predicts the marginal persuasion of a controller fact.
- Excluded memories that already prove A0: 3,618 (task003 symmetric) and 23,310 (OP symmetric).

**Post-hoc (not pre-registered; `posthoc_identity.csv`):**

| setup | R², fact identity | R², fact identity × count |
|---|---|---|
| task003 symmetric | 0.38 | 0.55 |
| task003 no-proof | 0.78 | 0.82 |
| OP symmetric | 0.44 | 0.48 |
| OP no-proof | 0.51 | 0.57 |

**Reading:**
- Which controller fact is added explains most of the variation. The agent's count adds a noticeable part only in
  task003 symmetric.
- A single strength per class is the main omission of both models: controller facts are heterogeneous in
  persuasiveness.
- This is the specific failure that the plan named as the justification for a trait density (evidence strength θ) for
  facts.
