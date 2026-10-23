# Scientific review of M1–M4 and a theory-first continuation

**Purpose:** Review the current theoretical results, critically evaluate the proposed extinction-based continuation, and specify a minimal set of calculations capable of explaining both principal phenomena. The final section presents an **alternative continuous-time, two-reservoir theoretical direction**.

**Primary source:** `theory_review_report.md` (22 October 2026), *Theory of the reading reversal and persistence: models M1–M4, scaling, and a second task*. Other quantities mentioned below are either defined here or explicitly identified as hypotheses, approximations, or proposed tests. The report's numerical evidence must not be presented as analytical proof.

**Audience:** The agent implementing the simulator and developing the theory. Work from the existing reproducible code, data, preregistration, and decision log. Keep numerical facts, interpretations, untested hypotheses, and new predictions clearly separated.

---

## 1. Scientific target: explain **both** phenomena

The theoretical objective is **not** to obtain a better empirical surrogate for the simulator or to accumulate more heatmaps. We seek a minimal predictive explanation of the following two observations, under a common account of information circulation and evidence integration.

### Phenomenon I — memory-dependent reversal of communication's effect

With a false controller targeting wrong answer $A2$, define susceptibility at time $t$ by

$$
G(\rho,q,t)
=\mathbb E\big[x_{A2}^{C}(t)-x_{A2}^{S}(t)\big],
$$

where $C$ and $S$ indicate controller and silent runs, $\rho$ controls memory retention, and $q$ is the number of readable posts sampled per agent activation. For $q_2>q_1$, define

$$
D(\rho;q_1\to q_2,t)
=G(\rho,q_2,t)-G(\rho,q_1,t).
$$

At $t=30$, **$D>0$ means additional reading amplifies manipulation; $D<0$ means it protects the population.** Low-memory and high-memory agents often occupy different regimes. The crossover depends on the problem, proof availability, and the increment in $q$; it is not established to be a universal sharp phase transition.

### Phenomenon II — collective persistence after intervention ends

At $t_\star=30$, the controller stops. Nevertheless,

$$
\delta(t)=\mathbb E\big[x_{A2}^{C}(t)-x_{A2}^{S}(t)\big],\qquad t>t_\star,
$$

can remain nearly constant for a further 100 simulated time units—much longer than the lifetime of an unrefreshed individual fact. The four continuation branches show: sustained effect with normal communication; sustained effect even with a frozen readable board; rapid collapse with no reading/posting; and large reduction after removal of controller-exclusive facts. We need a theory that predicts **why information survives collectively, under which conditions it dies out, and what fraction of the behavioral response it maintains**.

A successful theory must link the two phenomena. Explaining survival without predicting the sign of $D$ is not sufficient.

---

## 2. What the current models actually establish

### 2.1 Simulator versus M4

The simulator represents $N=24$ explicitly interacting agents. At independent rate-one activations, each agent samples $q$ posts from the finite recent-board window, updates a forgettable memory, evaluates the *exact task-dependent Bayesian posterior*, probability-matches its vote, and selects one remembered fact to repost according to the simulator's evidence-dependent policy. The controller has its own gated, stochastic posting process.

M4 is a **stochastic reduced population model**, not an analytic solution. It retains the simulator's **exact posterior, voting, and posting routines**, plus a *single realised finite shared board* and stochastic integer counts of fact holders (including extinction). It replaces much of the original finite-agent joint dynamics with an approximate population representation and drops some correlations except those propagated by the common board.

The benefit of M4 is identifying **which finite-population effects cannot be averaged away**, while retaining quantitative predictive power. It does **not** yet explain the task-specific reading crossover independently of the microscopic Bayesian/posting rules it imports. Do not claim that its decisions have been derived from a small set of physical parameters.

### 2.2 Findings supported by the report

| Result | Observed evidence | Interpretation and limitation |
|---|---|---|
| Deterministic approximations M1/M2 fail | Held-out signs near chance; susceptibility errors $\approx0.15$–$0.18$ | A deterministic average board loses crucial finite-copy behavior; this does **not** prove extinction is the only missing ingredient. |
| M3 helps when extinction is restored | Held-out sign scores 13/17 (symmetric), 13/16 (no-proof); $G$ error $\approx0.12$ | Stochastic extinction matters, but does not fully account for magnitude or the amplifying side. |
| M4 improves greatly | Held-out $G$ error $0.027$, versus $0.12$ for M3; post-stop ordering 7/8 | The shared realised board and correlated fluctuations matter. Note M4 **does not improve the total counts** of resolved reading-increment sign matches over M3 (13/17 and 13/16), though the errors move and magnitude predictions improve. |
| Controller-only information is a persistence carrier | Correct qualitative responses to normal, frozen, no-communication and removal branches | Strong evidence for external-memory recycling. M4 residual magnitudes after fact removal are still imperfect. |
| Cross-task transfer is promising | Preregistered Object Placements: M4 $G$ error $0.030$ symmetric, $0.019$ no-proof; main $3\to6$ crossover correctly located or bracketed | Useful out-of-task prediction, especially compared with M3, but Object Placements **persistence has not yet been tested**. |
| Finite-size comparisons are confounded | Fixed-fact replication raises redundancy; $K$-project scaling changes individual fact informativeness and sometimes baseline evidence | These are scientifically informative *different scaling directions*, not a pure $N$-dependence test. |

The frozen-board experiment is particularly constraining: **ongoing agent reposting is not necessary once the board is externally held fixed**. Reading repeatedly from the fixed board is sufficient to maintain the effect. In ordinary continuation, however, reposting keeps controller facts in the sliding window. The theory must reproduce both, without treating the fixed-board condition as endogenous reproduction.

### 2.3 Causal inference: what is strong and what is not

The intervention preventing agents from forgetting their initial fact reduces susceptibility (at $\rho=0.25$) from $0.44$ to $0.09$ in the symmetric task and from $0.30$ to $0.20$ in no-proof. This supports a major protective role for original evidence. However, this intervention *also changes the amount and availability of evidence within each agent*, not solely the population-level extinction hazard. It does **not** isolate extinction from Bayesian evidence integration.

The difference M3 $\to$ M4 is suggestive of common-board correlations, **but M4 also treats the controller stochastically while earlier models do not**. Do not attribute the full quantitative improvement uniquely to board correlations without a matched ablation.

The fact that M4 can reproduce a phenomenon does not establish why that phenomenon occurs. The goal now is **a predictive condition**, with task-relevant quantities independently defined and estimated, for $D$ to change sign.

---

## 3. Assessment of the agent's proposed three-step continuation

**Decision:** Pursue the direction, but modify steps 1–3 to make the bridge from *fact survival* to *Bayesian evidence integration* explicit. Avoid an unconstrained search for new observables.

### Step 1 — quantify extinction using existing data: **yes, with specific controls**

The report and prior experiments already track total original-fact survival, controller-fact prevalence in memories, and controller-origin board occupancy. Thus the new goal is **not** merely to show again that controller facts circulate or that original facts are lost. Instead, test whether the proposed **competition between survival of original evidence and circulation of controller evidence** predicts the response to increasing $q$.

For each fact $f$, distinguish

$$
n_f(t)=\#\{i:f\in M_i(t)\},\qquad
k_f(t)=\#\{\text{readable board posts carrying }f\},
$$

and estimate the survival probability

$$
S_f(t;\rho,q)=\Pr\big[n_f(t)+k_f(t)>0\big].
$$

* For **agent-origin facts absent from the controller's pool**, $n_f=k_f=0$ is irreversible when no other external source exists.
* For **controller-exclusive facts**, study survival and extinction **after the controller stops**; while injection continues, the same fact can be reintroduced.
* Respect the distinction between *fact identities* and *copies*. If one post contains one fact, $k_f$ is its multiplicity among readable posts; verify implementation details.
* Compute paired estimates and uncertainty at a small number of prespecified representative settings on **both sides** of the reading crossover, in both tasks if memory trajectories exist. Reuse 0.25-time-unit memory snapshots, but note that these cannot resolve every intermediate acquisition, renewal, or extinction/reappearance event. Check event logs before inferring microscopic hazard rates.

**Crucial improvement:** separate *survival of a fact* from *preservation of informative fact combinations*. Counting 10 surviving irrelevant facts does not compensate for losing one indispensable member of a multi-fact proof. Do not infer the sign of $D$ from total survivor counts alone.

**Falsifiable test:** Does the proposed survival criterion, specified before comparing $D$, correctly order the two sides of the crossover? If it does not, report failure rather than inventing a weighted survival score after seeing the answer.

### Step 2 — infer a fact reproduction number: **yes, for persistence, not automatically for reversal**

Estimate from the simulator rules or recorded events the per-fact acquisition, forgetting, posting, and board-turnover processes. An effective reproduction number may indicate whether a rare fact can maintain a chain of copies following controller withdrawal.

Do **not** assume that the threshold for fact reproduction coincides with the threshold for the reading reversal:

$$
\mathcal R_f=1\quad\not\Longleftrightarrow\quad D(\rho;q_1\to q_2)=0.
$$

Why: increased reading can improve the persistence of **both** controller-origin and complementary original facts. The population's behavioral response depends on the joint information set and its posterior, not only on whether one selected fact propagates.

Avoid estimating a reproduction coefficient $\kappa_f$ by regressing against the observed $G$ or $D$. Its microscopic definition must arise from the reposting rule or independent propagation data. Account for conditioning on the agent's current evidence, because the probability of selecting $f$ is generally neither constant nor independent of other facts.

### Step 3 — finite-copy birth–death theory: **yes, but append an explicit evidence layer**

Use stochastic copying, forgetting, and extinction to predict **distributions of retained information**, not just their means. Then map those distributions to votes using a task-grounded approximation to Bayesian integration.

A purely neutral birth–death process can explain survival or extinction under restrictive assumptions. It generally **cannot** explain task-dependent crossover locations in a system whose reposting is selected by posterior beliefs and whose evidence contains non-additive logical dependencies.

The minimal successful theory must include:

1. Finite-copy stochastic fact dynamics, with a common finite board (or an analytical approximation explicitly justified by M4).
2. A tractable, *independently specified* evidence-integration mechanism that distinguishes proof-bearing from no-proof tasks.
3. Predictions for both $D(\rho;q_1\to q_2,30)$ and the post-stop $\delta(t)$ and continuation branches.

**Success criterion:** Crossovers and persistence timescales arise from propagating these equations **without fitting the observed $G$ or sign of $D$** for the target task. If the model succeeds on task003 but fails on Object Placements, expose the missing task-dependent statistic instead of fitting an arbitrary crossover parameter.

---

## 4. The essential missing connection: extinction $\longrightarrow$ Bayesian evidence

### 4.1 Why fact extinction probabilities alone are mathematically insufficient

Write $M_i(t)$ for the **joint set** of facts remembered by agent $i$, and

$$
\Phi_2(M)=P(A2\mid M)
$$

for the exact Bayesian response to a complete information set. For memory-state distributions $\mu_C(M;t)$ and $\mu_S(M;t)$, the belief-level susceptibility is

$$
\boxed{\displaystyle
G_{\mathrm{belief}}(t)
=\sum_M \big[\mu_C(M;t)-\mu_S(M;t)\big] \,\Phi_2(M).
}
$$

**This identity is the bridge the theory must approximate predictively.** A model of copying and forgetting supplies $\mu_C,\mu_S$; a model of evidence integration supplies $\Phi_2$. Their combination produces the response. The identity is *not* itself a theory; the scientific advance is a tractable predictive closure for the distribution and/or the response function.

Knowing only the separate survival probabilities $S_f$ or the separate occupancy means $m_f$ cannot reconstruct $\mu(M)$. Proofs often require **co-occurrence** of complementary facts in the *same* agent's memory. Two populations may have identical fact-by-fact survival but very different rates of proof completion.

A simple example is a proof requiring $\{f_1,f_2\}$. The marginal probabilities $P(f_1\in M)=P(f_2\in M)=1/2$ do not determine $P(\{f_1,f_2\}\subseteq M)$: depending on correlation, it can range from 0 to $1/2$. The aggregate amount of surviving information is identical, but proof availability differs radically.

Further, the model must retain the controller's effect on *original fact circulation*: $\mu_C$ is not obtained merely by taking $\mu_S$ and adding controller-only facts.

### 4.2 A minimal task-grounded evidence closure

Instead of allowing a general black-box function of all fact sets, identify the **few combinations that actually determine the answer**:

* **Proof-bearing tasks:** enumerate minimal sufficient proof sets (or a compact representation of their logical dependency structure). Define a proof-completion indicator
  $$
  z(M)=\mathbf 1\{M\text{ contains at least one sufficient proof of }A0\}.
  $$
  Track or approximate $P(z(M_i)=1)$ and its dependence on original-fact extinction and co-retention. The challenge is to derive co-retention from information dynamics, not to measure $z$ as another unexplained heatmap.
* **No-proof tasks:** represent the evidence that discriminates between $A0$ and $A2$ with a small set of *task-derived* evidence states, such as likelihood-ratio classes, and preserve the complementarity between facts. A proof-indicator-only model is insufficient here.
* In both cases, controller-origin evidence enters those evidence states and influences not only the posterior but also the *future selection of reposted facts*.

Construct the evidence map from the known symbolic world and exact Bayesian rule, **before** seeing the target response curves. Do not fit an arbitrary logistic function to $G(\rho,q)$ and call it evidence integration. Use the smallest summary that reproduces important posterior contrasts between fact combinations; report its approximation error independently from the population dynamics.

### 4.3 What should be predicted, quantitatively?

At minimum:

1. Fact survival / extinction curves and, critically, **probabilities of informative co-retention** for a few preregistered conditions.
2. Susceptibility curves $G(\rho,q,30)$ and sign/bracket of the reading increments $D$.
3. Persistence after $t=30$ under normal continuation, frozen board, no reading/posting, and removal of controller-only information.
4. Cross-task generalization (the distinct $3\to6$ crossover locations), without tuning the target task to its observed $D$.

For the actual **vote-based** $G$, account for asynchronous last-vote delay. The saved $t=30$ memory snapshots yield *current* Bayesian beliefs, whereas the observed vote was sampled during an earlier activation. If the reduced theory predicts only snapshot beliefs, describe the mismatch honestly; ideally augment it with the last-vote state or the distribution of time since the last activation. Do not treat equality between instantaneous belief susceptibility and vote susceptibility as exact.

---

## 5. Recommended research protocol: a bounded theory-building loop

### Phase A — exploit existing data, **without** a new exploratory sweep

1. Reproduce the M1–M4 scorecards and identify the **specific quantitative gaps**; audit the existing data/event logs for fact-wise survival and board counts.
2. Calculate the narrow extinction/co-retention tests in §3–4 for representative $\rho,q$ values; estimate uncertainty by **episode**, preserving controlled/silent and cross-$q$ pairing when verified.
3. Quantify how much of the reversal *cannot* be predicted from marginal fact survival alone. One useful counterexample is a pair of conditions with similar fact marginals but different joint proof-completion probability or opposite $D$ signs. Do not manufacture this comparison if the data do not contain it.
4. Establish the minimal symbolic evidence classes and a finite-copy mechanism *before* further tuning to the same $D$ grid.

### Phase B — derive and validate a predictive theory

1. Specify the state, stochastic transition rates, board approximation, evidence map, and free parameters in a concise mathematics note. Label every closure and explicitly state when it destroys correlations or makes extinction impossible.
2. Determine propagation parameters from simulator rules or **independent** microscopic observations; determine evidence parameters from the symbolic task, not from $G$.
3. Derive analytical consequences where possible (early-time growth, reproduction/extinction condition, metastable lifetime, proof-completion probabilities, or sign bound for $D$). An analytic **inequality or approximation with a stated domain** is more valuable than an untestable exact claim.
4. Predict data previously unused to determine the closure: task003 held-out cells, Object Placements, and post-stop branches. Show uncertainties, the crossover bracket, and *absolute* as well as sign errors.
5. If it fails, diagnose which assumption failed (board correlations, joint memory configurations, evidence closure, controller feedback, or finite-size noise); revise **one mechanism at a time**, and record the reason and effect. Do not keep adding observables or unconstrained fit coefficients until the model interpolates the simulator.

**Stop/decision rule:** After a small number of hypothesis-driven revisions, if a genuinely compact, predictive closure is not possible, report that fact. M4 already provides a credible stochastic mechanistic model; a partial analytical theory of persistence plus a validated explanation of the main reversal is better than a falsely closed universal theory.

### Phase C — smallest justified new simulations

Only perform simulations targeted at an identified missing test:

* **M3-to-M4 attribution:** match stochastic-controller treatment and mean board occupancy, while changing only whether agents read one shared realised board or independent board samples; this distinguishes shared correlations from controller stochasticity.
* **Second-task persistence:** if the theory predicts persistence on Object Placements *a priori*, run the four continuation branches there as a meaningful out-of-task test.
* **Event-level rates:** instrument fact acquisitions, reposts, and evictions only if saved snapshots and logs cannot estimate the needed transition rates. Snapshots spaced by 0.25 cannot identify events that begin and end between observations.
* **Finite-size effects:** defer new $N$ series until the theory identifies which of agent-to-fact ratio, board-to-agent ratio, evidence complexity, and controller pressure should be held fixed. Report each scaling direction separately rather than claiming a pure thermodynamic limit.

### Required agent deliverable

Produce a self-contained `theory_mechanism_and_validation.md` containing: assumptions and state variables; derivation; explicit connection between extinction and evidence integration; measurable predictions; all comparisons with confidence intervals; competing hypotheses and ablations; failures; scope/limitations; what is genuinely new beyond M4; and a short paper-ready explanation of the two phenomena. Keep scripts, outputs, parameter choices, and run manifests organized and reproducible. Update the decision log at every hypothesis revision.

---

## 6. Alternative theoretical direction: revive the simple continuous-time two-reservoir model **with stochastic extinction**

**Recommendation:** **Yes, investigate this direction as a controlled analytical approximation, not as a replacement for M4 in its naive deterministic form.** The M1/M2 failures directly warn that an ordinary differential equation for average occupancies cannot by itself reproduce finite-copy extinction. The right use of the ODE is to derive interpretable rates, fixed points, linear-stability conditions, and candidate scaling parameters, then supplement it with a finite-state stochastic process and an evidence closure.

### 6.1 Two reservoirs and two distinct loss processes

For fact $f$, define

$$
 m_f(t)=\Pr(f\in M_i(t))\quad\text{(population-average agent memory occupancy)},
 \qquad
 b_f(t)=\frac{\mathbb E[k_f(t)]}{W}\quad\text{(readable-board occupancy)}.
$$

Let $\gamma=-\ln\rho$ be the per-fact agent forgetting rate. Agents activate at rate one and draw $q$ posts without replacement from a board of size $W$. Conditional on $k_f$ posts carrying $f$, the probability of encountering $f$ at least once is

$$
 r_f(q,k_f)
 =1-\frac{\binom{W-k_f}{q}}{\binom{W}{q}},
$$

for valid $q\le W$ (take an impossible binomial coefficient as zero). A simple independent-sampling/large-$W$ approximation is $r_f(q,b_f)\approx 1-(1-b_f)^q$.

Define $S_f(t)$ as the probability that an activated **swarm agent** chooses fact $f$ for its post, averaged over current agent memories, Bayesian voting, the remembered support of candidate answers, and the *actual reading/posting policy*. Define $C_f(t)$ analogously for the controller conditional on a controller post. Let $\lambda_c(t)$ be its *effective* injection rate, including the gating and budget (not assumed constant during the run).

An interpretable **heuristic mean-field skeleton** is

$$
\boxed{\begin{aligned}
\dot m_f
&\approx (1-m_f)\,r_f(q,b_f)-\gamma m_f,\\[2pt]
\dot b_f
&\approx \frac{N}{W}\big(S_f-b_f\big)
+\frac{\lambda_c(t)}{W}\big(C_f-b_f\big).
\end{aligned}}
$$

The two forms of information loss are different:

* **Agent memory loss:** fact copies expire at rate $\gamma$; a later read can restore a forgotten copy.
* **Board loss:** posts leave the *last-$W$ window* through turnover, at a characteristic scale $W/(N+\lambda_c)$ when the board is continually updated. This is **not** exponential forgetting by the board itself; the ODE replaces FIFO eviction by a mean-turnover approximation.

Caveats: $S_f$ includes **selective posting**, not merely Bayesian inference; one post carries one selected fact; occupancies for different facts are coupled; reading excludes own posts in the simulator; the board is an ordered FIFO list, not an exactly memoryless reservoir. For exponential memory lifetimes, rereading an already stored fact does not change its instantaneous forgetting hazard, although it resets its expiration clock and a read can restore a missing fact.

### 6.2 What can be derived analytically?

After injection ends, approximate the behavior of a **rare fact** by

$$
 r_f(q,b_f)\simeq q b_f,\qquad S_f\simeq\kappa_f m_f,
$$

where $\kappa_f$ is a *microscopically defined*, evidence-dependent rare-fact reposting propensity, assessed at a specified surrounding evidence distribution. Then

$$
\frac{d}{dt}
\begin{pmatrix}m_f\\b_f\end{pmatrix}
\simeq
\begin{pmatrix}
-\gamma &q\\
\frac{N}{W}\kappa_f&-\frac{N}{W}
\end{pmatrix}
\begin{pmatrix}m_f\\b_f\end{pmatrix}.
$$

The zero-occupancy state is linearly unstable in this **deterministic approximation** when

$$
\boxed{\mathcal R_f^{\mathrm{MF}}=\frac{q\kappa_f}{\gamma}>1.}
$$

This threshold does **not** imply permanent survival of a fact in a finite stochastic swarm. With no external injection, $n_f=k_f=0$ is absorbing; the deterministic theory predicts growth when rare, whereas the finite stochastic theory must quantify establishment probability, extinction probability, and finite-time lifetime. At $\rho=1$ ($\gamma=0$), use the original equations rather than dividing by $\gamma$.

Moreover, $\kappa_f$ changes as the collective evidence changes. A controller-only fact can proliferate and simultaneously lose persuasive power when agents co-retain proof-relevant original facts. Accordingly, **$\mathcal R_f^{\mathrm{MF}}=1$ is a propagation threshold, not a general sign criterion for $D$.**

### 6.3 Minimal stochastic extension (essential given M1–M4)

Replace $(m_f,b_f)$ by integer copies $(n_f,k_f)$, $0\le n_f\le N$, $0\le k_f\le W$. For a simplified well-mixed memory-acquisition process, candidate rates are

$$
\begin{aligned}
(n_f,k_f)\to(n_f+1,k_f)&:\quad (N-n_f)\,r_f(q,k_f),\\
(n_f,k_f)\to(n_f-1,k_f)&:\quad \gamma n_f.
\end{aligned}
$$

Board arrivals arise from swarm posts at rate $N S_f$ and controller posts at rate $\lambda_c C_f$; departures must be treated with **actual FIFO order** for fidelity to M4 or explicitly approximated by random replacement at rate $(N+\lambda_c)k_f/W$. For the exact simulation, an agent-specific read probability is required because a reader excludes its own board posts. The integer count model, especially with a shared stochastic board, makes simultaneous global absence $n_f=k_f=0$ a real absorbing state after injection stops.

The challenge is to close $S_f$: it depends on the **joint memory/evidence states** of agents and on correlations created by the board. A model tracking only $(n_f,k_f)$ for each fact may be insufficient even after adding extinction. The smallest additional state should encode the co-retention of proof-critical and conflicting evidence, not an arbitrary list of descriptive metrics.

### 6.4 Required bridge to Bayesian integration in this alternative

Specify $\Phi_2(M)=P(A2\mid M)$ (exact or principled coarse-graining from the symbolic task), and derive a predictive approximation to the controlled and silent joint memory distributions $\mu_C(M;t)$ and $\mu_S(M;t)$. Then compute

$$
G_{\mathrm{belief}}(t)
\approx \sum_M[\mu_C(M;t)-\mu_S(M;t)]\Phi_2(M),
\qquad
D_{\mathrm{belief}}=\Delta_q G_{\mathrm{belief}}.
$$

Equivalently, a very compact approximation may track a handful of evidence states (e.g., proof completed / incomplete, strength of opposing evidence), provided its transitions are **derived from information dynamics and task structure**. Do **not** label $S_f$ an explanatory closure while continuing to evaluate it through the full original simulator and presenting the result as an independent analytic theory.

**Discriminating tests for this alternative:**

1. Check whether the deterministic stability estimate predicts which facts tend to proliferate early, while the finite-copy extension predicts extinction times and post-stop survival more accurately.
2. Test whether incorporating *joint evidence states* shifts the predicted reading-reversal boundary toward the observed values relative to survival-only models.
3. Reproduce the four continuation branches by changing the actual mechanisms: stop injection; pin board content; set reading/posting to zero; or delete controller-exclusive copies. Do not fit each branch separately.
4. Predict the different task003 and Object Placements crossover brackets **using the same propagation structure** but task-derived evidence maps. If successful, this would connect finite-population information ecology to rational evidence integration in a transparent, paper-worthy theory.

### 6.5 Choice between theoretical routes

Do **not** run M4, a stochastic two-reservoir model, and multiple alternative closures as unlimited parallel projects. Treat the two-reservoir equations as a **specific candidate explanation**:

* Begin with a short analytic derivation and an extinction-versus-evidence diagnostic from existing data.
* Promote the model only if its state and independently measured parameters are meaningfully simpler than M4 and it yields **new, testable predictions**.
* Otherwise retain M4 as the validated stochastic surrogate and report the ODE as a controlled approximation explaining early propagation and the limitations of deterministic closure.

**Central criterion:** We understand the two phenomena only when a compact set of independently grounded assumptions jointly predicts **what information survives** and **what the surviving information causes agents to believe and post**.
