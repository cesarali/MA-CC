# Task: Implement a Binary Committee-Empowerment Probe for the Naming Game

## Objective

Extend the existing naming-game implementation with a **separate, reusable empowerment probe**. The naming game itself should remain unchanged except for explicit committee interventions controlled by configuration. Complete the task in one pass: add the experiment configurations, history logging, empowerment estimator, null tests, provider abstraction, runnable entry point, and basic tests.

The first version uses a two-name game, \(W=2\), because the resulting discrete channels can be estimated transparently by direct counting. Do **not** use embeddings, InfoNCE, or an LLM classifier.

---

## 1. Experiment layer

Create a configuration-driven experiment runner, separate from the information-theoretic analysis. Reuse the current naming-game code where possible.

### Required fixed configuration

```yaml
population_size: 24
names: [A, B]
memory_length: 5
max_population_rounds: 30
committee_sizes: [0, 1, 2, 3, 4, 6, 8]
replications_per_cell: 100
temperature: <existing default>
seed: <base seed>
```

A population round must be defined consistently with the existing implementation and recorded both as population rounds and raw pairwise interactions.

### Required regimes

1. **Neutral-start policy selection**
   - Start with empty memories.
   - Randomize the episode-level committee policy
     \(G \in \{	ext{always-A},	ext{always-B},	ext{no-committee}\}\).
   - Purpose: estimate whether the committee selects among alternative terminal conventions.

2. **Consensus-attack / resilience**
   - Initialize the ordinary population at complete consensus on A or B, with corresponding successful memories.
   - Assign the committee to support the incumbent, promote the alternative, or remain absent.
   - Run both directions: strong-to-weak and weak-to-strong whenever the existing code identifies a strong/weak convention.
   - Purpose: reproduce critical-mass asymmetry and compare takeover with empowerment.

3. **Pulse intervention**
   - Begin from consensus.
   - Activate the alternative-promoting committee for a configurable duration `pulse_rounds`, then restore committee members to ordinary-agent behaviour or remove the intervention.
   - Sweep at least `pulse_rounds: [1, 3, 5, 10]`.
   - Purpose: distinguish transient influence, delayed amplification, permanent flips, and recovery.

All model, prompt, payoff, partner-sampling, and ordinary-memory rules must remain identical across matched cells. Randomize committee identities and policy assignments by episode.

---

## 2. History and storage contract

Introduce a game-agnostic trajectory schema. The probe must operate **only on stored histories**, without calling an LLM or rerunning the simulation.

Store one row per interaction:

```text
episode_id, seed, regime, provider, model, prompt_hash,
N, W, H, committee_size, committee_ids, committee_policy,
pulse_active, interaction_index, population_round,
agent_i, agent_j, output_i, output_j,
success, payoff_i, payoff_j,
memory_i_before, memory_j_before,
share_A_recent, macrostate_binary, terminal_outcome
```

Use JSONL or Parquet. Also store one episode-summary row containing initial condition, stopping time, final convention, unresolved flag, takeover flag, recovery time, and total committee actions.

Define the recent production share using a configurable rolling window, default `window_interactions = 3 * N`.

Primary binary macrostate:

\[
Z_t = \mathbf 1[p_A(t) \ge 0.5].
\]

For exact ties, carry forward the previous non-tied macrostate; if none exists, mark the row unavailable for lagged analysis. Add a sensitivity representation with three states:

- B-dominant: \(p_A < 0.4\)
- mixed: \(0.4 \le p_A \le 0.6\)
- A-dominant: \(p_A > 0.6\)

Terminal outcome should be `{A, B, unresolved}`; additionally provide the binary `{A, B}` analysis restricted to resolved episodes.

---

## 3. Reusable probe module

Implement a standalone module, for example:

```text
analysis/
  empowerment.py
  estimators.py
  surrogates.py
  metrics.py
```

The public API should accept a dataframe/history path plus column mappings, so it can be reused for other discrete games.

### Primary quantities

For every committee size and experimental stratum, estimate:

\[
E_{\mathrm{terminal}}(k)=I(G;Y_T\mid k)
\]

and, for configurable horizons \(\ell\),

\[
E_{\mathrm{lag}}(k,\ell)=I(G;Z_{t+\ell}\mid Z_t,k).
\]

Use direct contingency-table estimation. Add **Jeffreys smoothing** of \(+1/2\) to every possible cell before normalization. Provide an unsmoothed estimate and a Miller–Madow-corrected sensitivity estimate. Never silently discard zero cells or unresolved episodes.

Treat episodes—not time rows—as independent units. Produce episode-bootstrap confidence intervals.

### Required complementary metrics

For each \(k\) and regime, report:

- takeover probability;
- terminal convention probabilities;
- consensus/stopping time;
- peak displacement in \(p_A(t)\);
- time to peak;
- recovery time after pulse removal;
- permanent-flip probability;
- number of committee actions;
- efficiency: terminal empowerment divided by expected committee actions.

### Required null tests

1. Shuffle `committee_policy` across complete episodes within the same regime, committee size, model, prompt version, and initial-condition stratum.
2. Circularly shift each episode’s macrostate trajectory for the lagged statistic.
3. Swap A/B labels in half the episodes and verify invariance.
4. Confirm near-zero empowerment in the no-committee baseline.

Return tidy result tables and create these plots:

1. terminal empowerment vs. committee fraction;
2. takeover probability on the same committee-size axis;
3. lagged empowerment vs. horizon;
4. mean \(p_A(t)\) trajectories during and after pulse intervention;
5. recovery time and empowerment efficiency vs. committee size.

---

## 4. LLM provider abstraction

Keep provider code independent of the game logic. Define a common interface such as:

```python
class LLMProvider:
    def generate(self, messages, *, temperature, max_tokens, seed=None) -> str:
        ...
```

Retain the existing university provider and add an OpenAI implementation selected through configuration:

```yaml
provider: university   # or openai
model: <provider-specific model>
fallback_provider: openai
allow_fallback: false
```

Use environment variables for credentials. Do not silently switch providers: fallback must be explicitly enabled, and every episode must log the actual provider and model used. The game runner must otherwise receive identical prompts and parsing logic.

---

## Acceptance criteria

- One command runs any configured regime and writes complete histories.
- A second command analyzes existing histories without simulation access.
- Re-running analysis on the same history is deterministic.
- Unit tests verify MI on independent, perfectly controlled, and constant-input synthetic channels.
- Tests verify Jeffreys smoothing, label-swap invariance, episode-level shuffling, and provider selection.
- Existing naming-game behaviour remains reproducible when committees and probes are disabled.
