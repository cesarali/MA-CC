# Ashery–Baronchelli Naming-Game Metrics

This document specifies the two trajectory metrics used to describe convention formation in the Ashery–Baronchelli naming-game experiments:

1. **Success rate**
2. **Production probability**

The definitions below are intended to be implemented in a multi-agent-system repository. They operate on stored pairwise interaction records and should remain independent of the LLM provider and game engine.

---

## 1. Required interaction data

Each pair interaction \(t\) must record at least:

```text
interaction_index
player_1_id
player_2_id
player_1_action
player_2_action
```

For committed-minority experiments, also record:

```text
player_1_is_committed
player_2_is_committed
```

Let

\[
a_t^{(1)}, a_t^{(2)} \in \mathcal{W}
\]

be the two names produced during interaction \(t\), where \(\mathcal{W}\) is the set of available names.

Each interaction therefore contributes:

- **one pair outcome** for the success-rate calculation;
- **two individual action productions** for the production-probability calculation.

These two denominators must not be confused.

---

## 2. Success rate

### 2.1 Per-interaction success indicator

An interaction succeeds exactly when the two agents produce the same name:

\[
y_t =
\mathbf{1}\!\left[a_t^{(1)} = a_t^{(2)}\right].
\]

Equivalently,

\[
y_t =
\begin{cases}
1, & a_t^{(1)} = a_t^{(2)},\\
0, & a_t^{(1)} \neq a_t^{(2)}.
\end{cases}
\]

The selected name does not matter. Both \((M,M)\) and \((Q,Q)\) are successes.

### 2.2 Success rate over a bin or window

For a set \(B\) containing \(L\) pair interactions, define

\[
\operatorname{SuccessRate}(B)
=
\frac{1}{L}
\sum_{t\in B}
\mathbf{1}\!\left[a_t^{(1)} = a_t^{(2)}\right].
\]

In words:

```text
success rate =
    number of interactions in which the two actions match
    ------------------------------------------------------
               number of pair interactions
```

The result lies in

\[
0 \leq \operatorname{SuccessRate}(B) \leq 1.
\]

### 2.3 Example

Suppose a bin contains four interactions:

```text
(M, M)
(M, Q)
(Q, Q)
(M, M)
```

The corresponding success indicators are

```text
1, 0, 1, 1
```

and therefore

\[
\operatorname{SuccessRate}
=
\frac{1+0+1+1}{4}
=
0.75.
\]

---

## 3. Production probability

### 3.1 Definition

For a particular name \(w\in\mathcal{W}\), the production probability is the fraction of **individual agent outputs** equal to \(w\).

For a bin \(B\) containing \(L\) pair interactions,

\[
\operatorname{ProductionProbability}(w;B)
=
\frac{1}{2L}
\sum_{t\in B}
\left(
\mathbf{1}\!\left[a_t^{(1)}=w\right]
+
\mathbf{1}\!\left[a_t^{(2)}=w\right]
\right).
\]

In words:

```text
production probability of name w =
    number of times w was produced by either player
    ------------------------------------------------
          total number of individual outputs
```

Because every ordinary pair interaction contributes two outputs, the denominator is \(2L\), not \(L\).

This is an empirical action frequency. It is **not** an LLM token probability, softmax probability, or model confidence.

### 3.2 Example

Using the same four interactions,

```text
(M, M)
(M, Q)
(Q, Q)
(M, M)
```

the eight individual outputs are

```text
M, M, M, Q, Q, Q, M, M
```

Hence,

\[
n_M=5,\qquad n_Q=3,
\]

and

\[
\operatorname{ProductionProbability}(M)=\frac{5}{8}=0.625,
\]

\[
\operatorname{ProductionProbability}(Q)=\frac{3}{8}=0.375.
\]

For a complete action pool,

\[
\sum_{w\in\mathcal{W}}
\operatorname{ProductionProbability}(w;B)
=
1.
\]

---

## 4. Why the metrics are different

The success rate is a statistic of **pairs**:

\[
\Pr\!\left(a_t^{(1)}=a_t^{(2)}\right).
\]

The production probability is a statistic of **individual outputs**:

\[
\Pr(a=w).
\]

Two trajectories can have identical production probabilities but very different success rates.

### Perfect coordination

```text
(M, M)
(M, M)
(Q, Q)
(Q, Q)
```

Then

\[
P(M)=0.5,
\qquad
\operatorname{SuccessRate}=1.
\]

### Complete mismatch

```text
(M, Q)
(M, Q)
(Q, M)
(Q, M)
```

Again,

\[
P(M)=0.5,
\]

but now

\[
\operatorname{SuccessRate}=0.
\]

Therefore, production probability alone does not measure coordination.

---

## 5. Binning into population rounds

In the Ashery–Baronchelli setup, a population round is operationally represented by \(N\) pair interactions, where \(N\) is the population size:

\[
1\text{ population round}=N\text{ pair interactions}.
\]

For population size \(N\), non-overlapping population-round bins can be defined as

\[
B_r
=
\{rN,\ldots,(r+1)N-1\}.
\]

For each bin \(B_r\), compute:

\[
S_r=\operatorname{SuccessRate}(B_r)
\]

and, for every name \(w\),

\[
P_r(w)=\operatorname{ProductionProbability}(w;B_r).
\]

Important: \(N\) pair interactions contain \(2N\) participation slots. Random pair sampling does **not** guarantee that every agent participates exactly once in a population round.

The repository should make the binning mode explicit. Recommended configuration:

```yaml
metrics:
  bin_size_interactions: ${population_size}
  partial_final_bin: drop
```

Possible policies for an incomplete final bin are:

- `drop`: omit it;
- `include`: compute the metric using the available interactions;
- `error`: reject trajectories whose lengths are not divisible by the bin size.

For reproducible paper-style comparisons, use one fixed policy across all runs.

---

## 6. Committed-minority experiments

When measuring whether ordinary agents adopt the minority convention, forced actions produced by committed agents should not automatically inflate the production probability.

For a name \(w\), define the ordinary-agent production probability as

\[
P_{\mathrm{ordinary}}(w;B)
=
\frac{
\text{number of non-committed outputs equal to }w
}{
\text{number of all non-committed outputs}
}.
\]

Filtering is applied **per output**, not per interaction. For example, if one participant is committed and the other is ordinary, retain the ordinary participant's output and exclude only the committed participant's output.

The success rate can still be computed over the actual pair outcomes unless the experimental analysis explicitly defines a different population subset.

Recommended metric configuration:

```yaml
production_probability:
  exclude_committed_outputs: true
```

This option should default to `false` outside committed-minority analyses.

---

## 7. Consensus detection is a separate use of success rate

The paper's committed-minority analysis uses a rolling success criterion to detect a consensus flip:

\[
\frac{1}{3N}
\sum_{s=t-3N+1}^{t}
\mathbf{1}\!\left[a_s^{(1)}=a_s^{(2)}\right]
\geq 0.95.
\]

Thus, a flip is detected when at least \(95\%\) of the most recent \(3N\) pair interactions are successful.

This rolling criterion must not be confused with the non-overlapping bins used to plot the trajectory. Both use the same per-interaction success indicator, but they use different windows:

```text
trajectory plot:
    non-overlapping bins, commonly N interactions each

consensus detection:
    rolling window of 3N interactions
```

If the implementation also needs to determine **which** convention has won, success rate alone is insufficient. It should additionally inspect the production probability or action share within the same window.

---

## 8. Reference implementation

```python
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Iterable, Sequence


Action = Hashable


@dataclass(frozen=True)
class InteractionRecord:
    interaction_index: int
    player_1_action: Action
    player_2_action: Action
    player_1_is_committed: bool = False
    player_2_is_committed: bool = False


def success_rate(records: Sequence[InteractionRecord]) -> float:
    """Fraction of pair interactions in which both actions match."""
    if not records:
        raise ValueError("success_rate requires at least one interaction")

    successes = sum(
        record.player_1_action == record.player_2_action
        for record in records
    )
    return successes / len(records)


def production_probabilities(
    records: Sequence[InteractionRecord],
    *,
    action_space: Iterable[Action] | None = None,
    exclude_committed_outputs: bool = False,
) -> dict[Action, float]:
    """
    Empirical distribution of individual agent outputs.

    Each ordinary pair interaction contributes two observations.
    If exclude_committed_outputs=True, committed-agent outputs are removed
    individually from both the numerator and denominator.
    """
    counts: Counter[Action] = Counter()
    total_outputs = 0

    for record in records:
        outputs = (
            (record.player_1_action, record.player_1_is_committed),
            (record.player_2_action, record.player_2_is_committed),
        )

        for action, is_committed in outputs:
            if exclude_committed_outputs and is_committed:
                continue
            counts[action] += 1
            total_outputs += 1

    if total_outputs == 0:
        raise ValueError("No eligible outputs for production probability")

    if action_space is None:
        actions = tuple(counts)
    else:
        actions = tuple(action_space)

    return {
        action: counts[action] / total_outputs
        for action in actions
    }


def non_overlapping_bins(
    records: Sequence[InteractionRecord],
    *,
    bin_size: int,
    include_partial_final_bin: bool = False,
) -> list[Sequence[InteractionRecord]]:
    if bin_size <= 0:
        raise ValueError("bin_size must be positive")

    bins: list[Sequence[InteractionRecord]] = []

    for start in range(0, len(records), bin_size):
        current = records[start : start + bin_size]
        if len(current) < bin_size and not include_partial_final_bin:
            break
        bins.append(current)

    return bins
```

---

## 9. Recommended output schema

Store long-form metric records rather than embedding every action as a separate column.

### Success rate

```text
episode_id
bin_index
start_interaction
end_interaction
num_pair_interactions
success_count
success_rate
```

### Production probability

```text
episode_id
bin_index
start_interaction
end_interaction
action
eligible_output_count
action_count
production_probability
excluded_committed_outputs
```

Keeping raw counts alongside normalized values allows the metrics to be validated and recomputed later.

---

## 10. Required validation tests

### Success rate

1. All pairs match \(\Rightarrow\) success rate is \(1\).
2. No pairs match \(\Rightarrow\) success rate is \(0\).
3. Three matches among four interactions \(\Rightarrow\) success rate is \(0.75\).
4. Empty input raises an explicit error.

### Production probability

1. Probabilities sum to \(1\) over the complete action space.
2. A bin with \(L\) ordinary interactions has \(2L\) eligible outputs.
3. Unobserved legal actions receive probability \(0\) when an action space is supplied.
4. Excluding committed outputs changes both numerator and denominator.
5. If all outputs are excluded, raise an explicit error instead of returning `NaN`.

### Joint sanity check

For

```text
(M, M), (M, Q), (Q, Q), (M, M)
```

the implementation must return

```text
success_rate = 0.75
production_probability[M] = 0.625
production_probability[Q] = 0.375
```

---

## 11. Implementation invariants

The implementation should preserve the following distinctions:

- Success rate counts **pair interactions**.
- Production probability counts **individual outputs**.
- One pair interaction normally contributes one success indicator and two action observations.
- Production probability is an empirical frequency, not a provider probability.
- Metric calculation should consume persisted interaction records and should not call an LLM.
- Binning, rolling windows, committed-output filtering, and incomplete-bin policy should be explicit configuration.
- Raw counts should be stored with normalized metrics.
- Consensus detection should not be inferred from production probability alone or from success rate alone without identifying the dominant action.
