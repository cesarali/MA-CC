# Empowerment MI — closed forms for the synthetic games

*05.08.2026 — companion to `05082026_synthetic_games_plan.md`*

## Goal

The system computes two families of mutual information. Family 1 (within-episode
`I(A_i;A_j)`) has an answer key. Family 2 — the "empowerment" statistics, `I(C;O)` and
`I(C;S_{t+h}\mid S_t)` — currently does not, and is estimated but never checked.

This document supplies the answer key for family 2. Everything below is exact: closed-form
where the algebra closes, exact linear algebra otherwise. No Monte Carlo, no approximation.

---

## 1. Why the "no closed form" argument fails

The stated reason family 2 can't be checked:

> empowerment isn't a property of a config in isolation, it's a property of which
> conditions you chose to sweep and how many episodes you ran per cell — an
> experimental-design choice, not a fixed fact about `epsilon` and `population_size`.

Both clauses are true. The conclusion doesn't follow.

The sweep grid **is** the input distribution $p(c)$ of a channel. Equal repetitions per
cell means $p(c)$ is uniform over the grid; unequal reps means $p(c)$ is proportional to
reps. Either way it is a distribution **we chose and know exactly**. Given it,

$$I(C;O) = H(O) - \sum_c p(c)\,H(O \mid c)$$

and $p(o\mid c)$ is fixed by dynamics we wrote. "Depends on a design choice" and "is not
analytically computable" are different claims. Condition on the design — which we always
can, because we made it — and the quantity is fully determined.

**The real obstacle was architectural, not mathematical.** Family 2's ground truth doesn't
live on the game object, because a single `BernoulliGame` instance doesn't know what it's
being swept against. It lives one level up, at the sweep. There is currently no object at
that level, which is why it looked impossible. The fix is to create one — see §6.

---

## 2. Notation

| Symbol | Meaning |
|---|---|
| $C$ | swept condition (the grid cell), with design distribution $p(c)$ |
| $N$ | population size |
| $\varepsilon$ | per-agent flip probability |
| $K$ | number of agents playing $Q$ in a round |
| $M = \varphi(x)$ | macrostate — the analysed coarse-graining of the microstate $x$ |
| $O$ | terminal outcome — dominant action in the final round |
| $h$ | horizon |
| $T_c$ | microstate transition matrix under condition $c$ |

---

## 3. Game 1 — Bernoulli

Agents draw $A_i = Z \oplus B_i$ with $Z_t \sim \mathrm{Bern}(1/2)$ fresh each round and
$B_{i,t} \sim \mathrm{Bern}(\varepsilon_i)$ private. Rounds are i.i.d. given $c$.

### 3.1 Terminal MI is exactly zero

Given $Z=0$, the count playing $0$ is $\mathrm{Bin}(N, 1-\varepsilon)$. Given $Z=1$, the
count playing $1$ has the same law. So $P(D=Q\mid Z=1) = P(D=M\mid Z=0) = p_{\text{maj}}$, and

$$P(D=Q) = \tfrac12 p_{\text{maj}} + \tfrac12\left(1-p_{\text{maj}}\right) = \tfrac12$$

for **every** $\varepsilon$ and **every** $N$. The terminal marginal is uniform regardless
of condition, therefore

$$\boxed{\;I(C;O_{\text{terminal}}) = 0 \quad \text{exactly, for any sweep.}\;}$$

This is a structural zero covering the entire terminal statistic on Game 1. Pass/fail, not
close/not-close. It is the single cheapest check in the whole harness.

### 3.2 The tie-break trap — and its closed form

The symmetry argument assumes no ties. With **even $N$**, ties occur, and if
`_dominant_action` resolves them by `argmax` label order they all go to the same action.
Then the symmetry breaks:

$$P(O = Q \mid c) = \tfrac12 + \tfrac12\, p_{\text{tie}}(c), \qquad
p_{\text{tie}}(\varepsilon, N) = \binom{N}{N/2}\bigl[\varepsilon(1-\varepsilon)\bigr]^{N/2}$$

(both mixture components contribute equally, so the $\tfrac12$'s cancel). Since
$p_{\text{tie}}$ depends on the swept parameter, the outcome marginal now varies with
condition and the pipeline reports **nonzero empowerment produced entirely by a tie-break
convention**. Its exact magnitude:

$$I(C;O) = H(\bar{p}) - \sum_c p(c)\, H\!\left(\tfrac12 + \tfrac12 p_{\text{tie}}(c)\right),
\qquad \bar{p} = \sum_c p(c)\left(\tfrac12 + \tfrac12 p_{\text{tie}}(c)\right)$$

**Use this deliberately.** Run odd $N$ to confirm the exact zero; run even $N$ to confirm
the pipeline produces *precisely* this spurious value and no more. If it produces something
else, the bug is in `reader.py`, not the estimator — and in a real run it would be
invisible.

### 3.3 Lagged conditional MI — closed form, flat in $h$

Given $c$, rounds are independent, so $S_{t+h} \perp S_t \mid C$. But $S_t$ and $S_{t+h}$
are *not* unconditionally independent — they share $C$ as a common cause. Working it
through:

$$I(C; S_{t+h}\mid S_t) = \underbrace{H(S) - H(S\mid C)}_{I(C;S)} \;-\; I(S_t; S_{t+h})$$

$$\boxed{\;I(C;S_{t+h}\mid S_t) = I(C;S) - I(S_t;S_{t+h}), \quad \text{independent of } h.\;}$$

with the macrostate law a symmetric binomial mixture:

$$p(k\mid c) = \binom{N}{k}\cdot\tfrac12\left[(1-\varepsilon_c)^k \varepsilon_c^{N-k} + \varepsilon_c^k(1-\varepsilon_c)^{N-k}\right]$$

and $I(S_t;S_{t+h}) = \sum_{k,k'} p(k,k')\log\frac{p(k,k')}{p(k)p(k')}$ where
$p(k,k') = \sum_c p(c)\,p(k\mid c)\,p(k'\mid c)$. All finite sums over $k = 0\ldots N$.

**The diagnostic is the shape, not the number.** The curve must be *flat* in $h$ at a
height you predicted in advance. Any slope is an artifact — windowing, episode-boundary
edge effects, or the circular-shift null leaking into the estimate. A flat line of known
height is a far stronger check than a single value.

### 3.4 Serious warning: never sweep `population_size` against a raw macrostate

If the condition is $N$ and the macrostate is `population_action_share_per_option`, the
support of the share is $\{0, 1/N, 2/N, \ldots, 1\}$ — **a different grid for each $N$**.
Observing a share of $0.15$ identifies $N=20$ with certainty; it is not attainable at
$N=10$.

The condition is then near-perfectly recoverable from a single macrostate observation, and
the pipeline will report $I(C;S) \approx H(C)$ — maximal, and entirely an artifact of
alphabet support rather than of any dynamical influence.

Any sweep over $N$ must bin the share onto a **common grid shared across all conditions**
before the contingency table is built. This is worth checking in the existing code before
anything else, because if it's happening it invalidates every population-size result
already produced.

---

## 4. Game 2 — Markov

Microstate $x \in \{0,1\}^N$, so $|X| = 2^N \le 1024$ for $N \le 10$. Build $T_c$ from the
kernel, coupling graph, lags and $\varepsilon$.

### 4.1 Exact quantities

$$p_t(\cdot \mid c) = p_0 T_c^{\,t}$$

**Terminal:**

$$p(o\mid c) = \sum_x p_T(x\mid c)\,\mathbb{1}\{\mathrm{dom}(x) = o\}$$

then $I(C;O)$ by the standard formula with $p(c)$ the design distribution.

**Lagged, at a single $t$:**

$$p(c, m, m') = p(c) \sum_{x:\varphi(x)=m} p_t(x\mid c) \sum_{x':\varphi(x')=m'} \left[T_c^{\,h}\right]_{xx'}$$

$$I(C;M'\mid M) = \sum_{c,m,m'} p(c,m,m')\,\log \frac{p(m'\mid m,c)}{p(m'\mid m)}$$

**If the pipeline pools over $t$ within an episode** (it does — pairs of current/future
macrostate per episode), average $p(c,m,m')$ over $t$ in the analysed window *before*
computing the CMI. Pooling then normalising is not the same as normalising then pooling;
match whatever `pipeline.py` actually does.

### 4.2 Symmetric dynamics give zero again

If the kernel and initial condition are both $Q/M$-symmetric, the absorbing consensus is
50/50 for every condition and $I(C;O_{\text{terminal}}) = 0$. A useful second null, useless
as a positive control. To get a nonzero terminal MI on Game 2 the symmetry must be broken
deliberately — asymmetric $p_0$, or a biased kernel.

This is diagnostic in itself: it says the config sweep **is not a control channel**.
$\varepsilon$ and $N$ change how noisy the population is, not which way it goes. That is
why empowerment-against-config keeps coming out at zero, and it is a fact about the
experiment, not a bug.

### 4.3 Lumpability — the trap

The macrostate is a coarse-graining. The macrostate process is Markov only under **strong
lumpability**, which holds under full exchangeability (identical $\varepsilon$, mean-field
pairing) and fails as soon as you use the asymmetric coupling graph or heterogeneous
$\varepsilon$.

When it fails, computing on the lumped $(N{+}1)$-state chain gives a **different and wrong**
answer versus propagating the full microstate distribution and lumping at the end. Only the
latter is correct. Always propagate in $2^N$ and lump last.

The gap between the two is itself computable, and it quantifies how much information the
macrostate is discarding. Worth reporting.

---

## 5. Game 3 — Controlled Markov (the positive control)

Sections 3.1 and 4.2 both give zero, which is the point: sweeping config parameters isn't
steering. Empowerment in the intended sense needs an input that actually moves the
population, and that is exactly $u$.

With $T_u$ the $u$-indexed family and $p(u)$ the design grid:

$$I(U;S_{t+h}) \quad\text{and}\quad I(U;S_{t+h}\mid S_t)$$

follow from §4.1 with $c := u$. Both genuinely nonzero, and monotone in control strength.

Two versions worth computing, because they answer different questions:

- **Design MI** — with $p(u)$ fixed to your grid. This is what the estimator is targeting,
  and the number it must reproduce.
- **Capacity** — $\max_{p(u)} I(U;S_{t+h})$, via Blahut–Arimoto over $\le 1024$ states,
  instant. This is the ceiling: how much control authority exists at all, independent of
  how you chose to sample it. Useful for knowing whether a low measured value means a weak
  controller or a badly chosen grid.

Sweeping $h$ gives a capacity-vs-horizon curve to check the estimator's *shape* against,
not just a point.

---

## 6. Implementation

### 6.1 Where it lives

Family 2's ground truth is **not** a method on the game. It needs the sweep grid, the
repetitions, the horizons and the binning. Add a sweep-level object:

```python
def sweep_ground_truth(grid_spec, analysis_spec) -> GroundTruth:
    """Exact family-2 quantities for a resolved sweep.

    grid_spec:     swept parameter, its values, repetitions per cell
                   -> gives p(c)
    analysis_spec: horizons, macrostate binning, analysis window
                   -> must match pipeline.py exactly
    """
```

Both arguments must be the **resolved objects that actually ran**, never hand-written
values. Otherwise the phantom bug returns: change the grid, forget the expected number,
lose a day debugging an estimator that was fine.

### 6.2 What it should return

```
terminal_mi_true          exact I(C;O)
lagged_cmi_true[h]        exact I(C;S_{t+h}|S_t) per horizon
alphabet_sizes            |C|, |M| after binning
n_effective               number of episodes (NOT rounds)
expected_plugin_bias      see below
lumpable                  bool — whether the macrostate chain is Markov
```

Shipping the bias estimate alongside the truth is what makes the comparison honest.

### 6.3 The bias that will dominate

Plug-in MI bias goes as $\mathrm{dof}/(2 N_{\text{eff}} \ln 2)$ bits, with

- terminal: $\mathrm{dof} = (|C|-1)(|O|-1)$
- lagged CMI: $\mathrm{dof} = |M|\,(|C|-1)(|M'|-1)$

$N_{\text{eff}}$ is the number of **episodes**, not rounds — which is why the bootstrap is
over `episode_id`. Within-episode pairs are correlated and don't buy independent samples.

Worked case: $N = 10$ agents gives $|M| = 11$ macrostates; with 2 conditions,
$\mathrm{dof} = 1 \times 10 \times 11 = 110$. At 100 episodes:

$$\frac{110}{2 \times 100 \times \ln 2} \approx 0.79 \text{ bits of pure bias.}$$

That will swamp any real effect. Jeffreys smoothing shrinks it but does not fix a
degrees-of-freedom problem. **Binning the macrostate to 3–5 levels is almost certainly not
optional.** The closed form is what lets you tell the difference between a real signal and
table inflation.

---

## 7. Validation order

Cheapest and most diagnostic first.

1. **Game 1 terminal MI, odd $N$.** True value is exactly 0. The observed estimate must sit
   inside the shuffle null. This validates `estimate_nulls` itself, because you know the
   truth — if the null doesn't cover a true zero, the null construction is broken and every
   significance claim downstream is unreliable.
2. **Game 1 terminal MI, even $N$.** Must reproduce the §3.2 tie-break value exactly.
   Catches `reader.py` contamination.
3. **Check the $N$-sweep binning** (§3.4). If shares aren't binned to a common grid, stop
   and fix before interpreting any population-size result.
4. **Game 1 lagged CMI.** Must be flat in $h$ at the predicted height. Catches windowing
   and horizon-alignment bugs.
5. **Game 2 lagged CMI**, propagated in $2^N$ and lumped last. Compare against the lumped-chain
   computation to quantify what the macrostate discards.
6. **Game 3 $I(U;S_{t+h})$ vs. exact, plus capacity.** The only real positive control — the
   one place a nonzero empowerment has a known target.

Steps 1–3 need no new estimator work and can run today against results you already have.
