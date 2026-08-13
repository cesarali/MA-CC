# Classical Kernel Choice — Round-Level Feedback Game

## Goal

Implement the provider-free classical kernel for:

```text
hidden_bench_imitation_round_feedback
```

Use a **unanimity q-voter / imitation rule**. The same focal response rule is used in ordinary and controlled microscopic updates. The controller changes only the composition of the social inputs.

This kernel does **not** impose microscopic reversibility.

---

## State

Let the answer options be:

```text
A = {a_1, ..., a_K}
```

and let the population occupation vector be:

\[
\mathbf n=(n_1,\ldots,n_K), \qquad \sum_j n_j=N.
\]

At every microscopic update:

1. sample one focal agent;
2. sample `q` distinct ordinary peers from the other `N-1` agents;
3. at most the focal opinion changes.

---

## Ordinary update: `K0`

The focal switches **only if all q ordinary social inputs unanimously support the same option and that option differs from the focal's current opinion**.

If the focal currently holds option \(i\), then for any \(j\neq i\),

\[
P(i\rightarrow j\mid \mathbf n,\text{ordinary})
=
\frac{\binom{n_j}{q}}{\binom{N-1}{q}}.
\]

Therefore the mesoscopic transition probability is

\[
K_0(\mathbf n+\mathbf e_j-\mathbf e_i\mid\mathbf n)
=
\frac{n_i}{N}
\frac{\binom{n_j}{q}}{\binom{N-1}{q}},
\qquad i\neq j.
\]

All remaining probability is a self-transition:

\[
K_0(\mathbf n\mid\mathbf n)
=
1-\sum_{i\neq j}
K_0(\mathbf n+\mathbf e_j-\mathbf e_i\mid\mathbf n).
\]

---

## Controlled update: `K1`

Let the controller target be \(Z\).

A controlled microscopic position contains:

```text
(q - 1) ordinary peer inputs + 1 controller input supporting Z
```

The focal uses the **same unanimity rule**.

Because one slot already supports \(Z\), the only possible controlled switch is:

```text
non-Z focal -> Z
```

and it occurs only if all remaining `q-1` ordinary inputs also support \(Z\).

For any focal option \(i\neq Z\),

\[
P(i\rightarrow Z\mid\mathbf n,\text{controlled})
=
\frac{\binom{n_Z}{q-1}}{\binom{N-1}{q-1}}.
\]

Hence

\[
K_1(\mathbf n+\mathbf e_Z-\mathbf e_i\mid\mathbf n)
=
\frac{n_i}{N}
\frac{\binom{n_Z}{q-1}}{\binom{N-1}{q-1}},
\qquad i\neq Z.
\]

No transition away from \(Z\) is allowed in a controlled slot under strict unanimity, because one social input is always the controller's \(Z\) advocacy.

Again,

\[
K_1(\mathbf n\mid\mathbf n)
=
1-\sum_{i\neq Z}
K_1(\mathbf n+\mathbf e_Z-\mathbf e_i\mid\mathbf n).
\]

### Important special case: `q = 1`

A controlled step has:

```text
0 ordinary peers + 1 controller slot
```

so every non-\(Z\) focal switches to \(Z\):

\[
P(i\rightarrow Z\mid q=1,\text{controlled})=1.
\]

A focal already at \(Z\) stays at \(Z\).

---

## Implementation contract

The classical runtime should receive the actual sampled social context and implement exactly:

```python
if not controlled_slot:
    # q ordinary peers
    if all effective inputs have the same option j and j != focal_opinion:
        new_opinion = j
    else:
        new_opinion = focal_opinion

else:
    # controller target Z + q-1 ordinary peers
    effective_inputs = [Z] + remaining_peer_opinions

    if all effective inputs have the same option Z) and focal_opinion != Z:
        new_opinion = Z
    else:
        new_opinion = focal_opinion
```

The implementation may evaluate the rule directly from the sampled peer opinions. The formulas above are the corresponding population-level transition probabilities used by the theory.

---

## Do not add in this first kernel

Do **not** add:

- logistic/soft social response;
- hidden `control_strength`;
- spontaneous opinion flips;
- anticonformity;
- detailed-balance corrections;
- extra controller decisions inside the round.

The controller policy and the preallocated `b`-position schedule are handled by the round-level game, not by this kernel.

---

## Validation tests

At minimum verify:

1. `K0`: a focal changes only under unanimous ordinary peers.
2. `K1`: a non-target focal changes to `Z` iff all `q-1` ordinary effective peers are `Z`.
3. `q=1`, controlled: every non-`Z` focal switches to `Z`.
4. `q=1`, ordinary: the focal copies the single peer if different.
5. Only the focal vote changes.
6. No provider calls occur in classical mode.
7. Empirical transition frequencies from many simulations match the analytical probabilities above.

---

## Theory connection

These microscopic kernels define the round kernels

\[
R_0(\mathbf n'\mid\mathbf n), \qquad
R_1(\mathbf n'\mid\mathbf n),
\]

where an `ADVOCATE` round contains exactly `b` preallocated `K1` positions and `N-b` `K0` positions.

The headline round-level transfer entropy is then

\[
T_{U\rightarrow N}
=
I(U_k;\mathbf N_{k+1}\mid\mathbf N_k).
\]

### Caveat

Strict unanimity dynamics can have absorbing consensus states. Do not silently add noise to remove them. If an ergodic reference kernel is needed later for a unique stationary distribution, implement that as a **separate explicit kernel variant**.
