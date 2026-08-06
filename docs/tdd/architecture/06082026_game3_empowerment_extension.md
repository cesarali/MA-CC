# Game 3 extension — empowerment as the paper defines it

*05.08.2026 — amendment to `05082026_empowerment_ground_truth.md` §5. Games 1–3 assumed already implemented.*

Reference: Song et al., *Estimating the Empowerment of Language Model Agents*, arXiv:2509.22504.
Effective empowerment is $\mathcal{E}(\pi) = \mathbb{E}\left[I(a_t; s_* \mid s_t)\right]$ with
$\tau \sim \mathrm{Geom}(1-\gamma)$ and the expectation taken over states encountered under the policy.

---

## 0. Notation — every symbol used below

| Symbol | Definition |
|---|---|
| $N$ | number of agents |
| $s \in \mathcal{S}$ | **microstate** — the full action profile of all $N$ agents, $\|\mathcal{S}\| = 2^N \le 1024$ |
| $m = \varphi(s)$ | **macrostate** — the coarse-grained summary actually used in `analysis/` (dominant-action-share bucket) |
| $u \in \mathcal{U}$ | **control input** — the committee's forced value; finite alphabet |
| $C$ | the episode's **condition** = grid cell = the value of $u$ assigned to that episode (used when $u$ is episode-fixed) |
| $T_u$ | environment kernel under control $u$: $T_u[s,s'] = P(s_{t+1}=s' \mid s_t=s,\ u_t=u)$ |
| $\pi(u\mid s)$ | control policy — how $u$ is chosen. Uniform/state-independent unless stated |
| $\bar T$ | **policy-averaged kernel**, $\bar T[s,s'] = \sum_u \pi(u\mid s)\, T_u[s,s']$ |
| $p_0$ | initial microstate distribution |
| $p_t$ | microstate distribution at round $t$ |
| $h$ | horizon in rounds |
| $\gamma$ | discount; $\tau \sim \mathrm{Geom}(1-\gamma)$, so $P(\tau = h) = (1-\gamma)\gamma^{h-1}$ |
| $d(s)$ | **state visitation distribution** — see §4, this is *not* the stationary distribution |
| $\mathcal{E}_h(s)$ | per-state empowerment at horizon $h$ |
| $\mathcal{E}$ | effective empowerment, the single scalar |

---

## 1. What needs to change and why

Current Game 3 draws $u$ once per episode and holds it fixed. That makes the macrostate $M_t$ a
**mediator** on the path $C \to M_t \to M_{t+h}$, so conditioning on $M_t$ blocks most of the
effect being measured. Symptoms: the measured CMI decays with $t$ for design reasons rather than
dynamical ones, and the reported value depends on which rounds were pooled.

The paper's $a_t$ is drawn *at* round $t$, so $s_t$ is pre-action and the conditioning is clean.

**Fix: keep both variants and measure the gap.**

| Variant | Control | Corresponds to |
|---|---|---|
| **3a** | $u$ drawn once per episode, held fixed | the current `mas_cc` control experiments |
| **3b** | $u_t \sim \pi(\cdot\mid s_t)$ resampled every round | the paper's $I(a_t; s_*\mid s_t)$ |

Both are exact on the finite chain. The difference between them stops being an argument and
becomes a number.

The algebraic statement of the difference: in **3a**, $p(s_t\mid u) \neq p(s_t)$ because $u$ has
been acting since round 0 — $s_t$ is causally downstream of $u$. In **3b**, $u_t$ has not yet
acted when $s_t$ is observed, so $s_t$ is not downstream of it.

---

## 2. Closed forms — 3b (per-round control, the paper's quantity)

Given $s_t = s$, applying $u_t = u$ once and then following the policy for $h-1$ further steps:

$$P(s_{t+h} = s' \mid s_t = s,\ u_t = u) = \left[T_u\, \bar T^{\,h-1}\right]_{ss'}$$

Marginalising the action recovers the policy-averaged $h$-step kernel exactly:

$$\sum_u \pi(u\mid s)\left[T_u \bar T^{\,h-1}\right]_{ss'} = \left[\bar T^{\,h}\right]_{ss'}$$

so the per-state empowerment is a policy-weighted KL divergence:

$$\boxed{\;\mathcal{E}_h(s) \;=\; \sum_u \pi(u\mid s)\; D_{\mathrm{KL}}\!\Big(\big[T_u \bar T^{\,h-1}\big]_{s\cdot} \;\Big\|\; \big[\bar T^{\,h}\big]_{s\cdot}\Big)\;}$$

$$= \sum_u \pi(u\mid s) \sum_{s'} \left[T_u\bar T^{\,h-1}\right]_{ss'} \log \frac{\left[T_u\bar T^{\,h-1}\right]_{ss'}}{\left[\bar T^{\,h}\right]_{ss'}}$$

Implementation is three matrix products and a KL per state. Nothing exotic.

**Start with a state-independent policy** $\pi(u\mid s) = \pi(u)$. Then $U_t \perp S_t$ exactly:
no mediator, no backdoor, and $\mathcal{E}_h(s)$ is pure control authority. Add state-dependence
only after that case is verified.

---

## 3. Closed forms — 3a (episode-fixed control, the current design)

$u$ drawn once from $p(u)$, then the chain is $T_u$ throughout.

$$p_t(s\mid u) = \left[p_0 T_u^{\,t}\right]_s, \qquad
P(s_{t+h}=s' \mid s_t=s,\, u) = \left[T_u^{\,h}\right]_{ss'}$$

$$p(u, s, s') = p(u)\; p_t(s\mid u)\; \left[T_u^{\,h}\right]_{ss'}$$

and $I(U; S_{t+h}\mid S_t)$ follows from that joint by the standard CMI sum. Note this is a
function of $t$ — that dependence is the contamination, and quantifying it is the point.

**Report $I^{3a}_h(t)$ as a curve in $t$, not one pooled number.** The exact computation predicts
monotone decay; if the estimator reproduces the exact decay, the pooling window is confirmed as
the dominant design knob and must be promoted from an incidental choice to a declared config
parameter.

**Gap to report:** $\Delta_h(t) = \mathcal{E}_h^{3b} - I^{3a}_h(t)$.

---

## 4. State distribution — the absorbing-chain trap

The paper averages over "states encountered under the policy." **Do not use the stationary
distribution.** The naming game is absorbing, so the stationary distribution sits entirely on
consensus states, where empowerment is zero — you would get $\mathcal{E} \approx 0$ trivially and
for the wrong reason.

Use the **visitation distribution** over the analysed window:

$$d(s) = \frac{1}{|W|}\sum_{t \in W} p_t(s)
\qquad\text{or, discounted:}\qquad
d_\gamma(s) = (1-\gamma)\sum_{t\ge 0}\gamma^t p_t(s)$$

Config must state which. If the two ever agree on this game, something is wrong.

---

## 5. The $\gamma$-weighted scalar

$$\mathcal{E} = \sum_{h\ge1}(1-\gamma)\,\gamma^{\,h-1} \sum_s d(s)\, \mathcal{E}_h(s)$$

Truncate at $H$ with residual mass $\gamma^H$; choose $H \ge \ln(\text{tol})/\ln\gamma$ and **log
the residual** so truncation error is visible rather than assumed.

Keep the per-horizon curve $\{\mathcal{E}_h\}$ as the primary output. The scalar is recoverable
from the curve; the curve is not recoverable from the scalar.

---

## 6. Per-state profile

Do not collapse to the average. The CMI is already $\sum_s d(s)\,\mathcal{E}_h(s)$ — report the
profile $\mathcal{E}_h(s)$, or its lumped version $\mathcal{E}_h(m)$, as a curve over macrostate.

Expected shape for the naming game: peaked near a 50/50 split, decaying to zero near consensus.
This is free (the terms are already computed) and is the analogue of the paper's "empowerment
traces highlighting influential states."

---

## 7. True state vs. macrostate

Compute both, on the same run:

$$I(U; S_{t+h}\mid S_t) \qquad\text{vs.}\qquad I(U; M_{t+h}\mid M_t), \quad M = \varphi(S)$$

The gap is the cost of coarse-graining, measured exactly. EELMA cannot do this — it uses InfoNCE
on embeddings precisely because it cannot enumerate states. Here we can.

**Warning:** there is no general inequality between the two. Coarse-graining the *conditioning*
variable can raise conditional MI as easily as lower it, so do not assume
$I(U;M'\mid M) \le I(U;S'\mid S)$ and do not use a violation as a bug signal.

Propagate in $\mathcal{S}$ ($2^N$) and lump last — see the lumpability note in the parent doc.

---

## 8. Config additions

```yaml
game_03:
  control_mode: per_round        # per_round (3b) | episode_fixed (3a)
  policy: uniform                # uniform (state-independent) | state_dependent
  control_alphabet: [Q, M]
  control_strength: 0.8

empowerment:
  gamma: 0.9
  horizon_tolerance: 1.0e-3      # sets H; residual gamma^H is logged
  state_distribution: visitation # visitation | discounted_visitation | stationary
  analysis_window: [0, 200]      # promoted from incidental to declared
  report_per_state_profile: true
  state_representation: [microstate, macrostate]   # compute both
```

---

## 9. Acceptance checks

1. **3b, $h=1$, uniform policy** — the KL form of §2 must agree with a brute-force enumeration of
   the joint $p(s,u,s')$ to machine precision.
2. **Marginalisation identity** — verify $\sum_u \pi(u\mid s)[T_u\bar T^{h-1}] = [\bar T^h]$
   numerically. Cheap, and catches kernel-construction errors immediately.
3. **3a decay** — $I^{3a}_h(t)$ decreases monotonically in $t$; the estimator reproduces the exact
   curve. If it doesn't decay, the kernel isn't actually being applied from round 0.
4. **Game 1 null** — no control input, so $\mathcal{E} = 0$ exactly at every horizon.
5. **Absorbing trap** — `stationary` and `visitation` must give different answers, with
   `stationary` ≈ 0. If they match, the visitation distribution is being computed wrong.
6. **Truncation** — logged residual $\gamma^H$ below `horizon_tolerance`.
7. **Zero-control limit** — set `control_strength: 0`, so $T_u = \bar T$ for all $u$ and
   $\mathcal{E}_h(s) = 0$ for all $s, h$.
