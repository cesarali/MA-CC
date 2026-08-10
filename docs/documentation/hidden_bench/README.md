# HiddenBench in `mas_cc`

Three games built on the Hidden Profile paradigm (Stasser & Titus, 1985) as
operationalized by Li, Naito & Shirado, *Systematic Failures in Collective
Reasoning under Distributed Information in Multi-Agent LLMs* (ICML 2026).

| Game | Module | Protocol |
| --- | --- | --- |
| `hidden_bench_vanilla` | [`games/hidden_bench/vanilla/`](../../src/mas_cc/games/hidden_bench/vanilla/) | The paper's own: N agents, round-robin plenary discussion, a vote before and after. |
| `hidden_bench_naming` | [`games/hidden_bench/naming/`](../../src/mas_cc/games/hidden_bench/naming/) | Dyadic: private pairs, private per-partner memory, on the expanded (N > 4) populations. |
| `hidden_bench_imitation` | [`games/hidden_bench/imitation/`](../../src/mas_cc/games/hidden_bench/imitation/) | One-focal opinion dynamics with matched LLM-reasoning and provider-free classical kernels plus partial-observation feedback control. |

The complete reference for the imitation game, including all metric
definitions, MI/CMI equations, estimator variants, report files, and executable
Conda commands, is
[`hidden_bench_imitation.md`](hidden_bench_imitation.md).

The corpus is **not** produced here — see
[`data_provenance.md`](data_provenance.md), which also records which parts of the
upstream preprocessing are still unfinished. For the complete generation,
verification, and population-allocation procedure behind the two semantic
scaling methods, see
[`paraphrase_and_factorization_pipeline.md`](paraphrase_and_factorization_pipeline.md).

---

## 1. The idea in one paragraph

Information `I` splits into **shared** `Is`, which everyone gets, and
**unshared** `Iu`, partitioned so agent `i` holds a unique `Iu_i`. Shared
information alone points at a wrong answer (the *decoy*), and so does shared
information plus any *single* hidden item; only pooling all the hidden items
disqualifies the decoy and identifies `o*`. Crucially, **agents are never told
the information is asymmetric.** The paper's finding is that groups reliably
fail to pool — not because they reason badly once they have the facts, but
because they never surface them. `unshared_disclosure_rate` is the metric that
separates those two explanations, and it is the one to look at first.

## 2. Running them

```bash
# Price it before you spend anything. Refuses to launch what it cannot price.
mas-cc experiment preflight --config configs/runs/hidden_bench_vanilla.yaml

mas-cc experiment run --config configs/runs/hidden_bench_vanilla.yaml
mas-cc experiment run --config configs/runs/hidden_bench_naming.yaml
mas-cc experiment run --config configs/runs/hidden_bench_grid.yaml     # a sweep
```

Both games are driven by `games/hidden_bench/runtime.py`, which reports every
step to the recorder — so every episode writes `metrics/streaming.csv`, and a
finished grid directory is readable by `mas-cc analysis empowerment --grid-dir`
without modification.

## 3. Config axes

### Shared by both games

| Option | Values | What it does |
| --- | --- | --- |
| `task_set` | `vanilla` \| `expanded` | The 65 upstream tasks, or a prebuilt N > 4 population. |
| `task_id` | task name | e.g. `evacuation_west_city`. Unset takes the first task. |
| `n_agents` | int | Must equal `game.population_size`; a disagreement is rejected at `initialize`. In a grid, sweep `game.population_size` and leave this unset. |
| `profile` | `hidden` \| `full` | `full` gives every agent all of `Iu` — the ceiling condition. **Not a separate game**, so `Y_full` and `Y_post` come from one code path. |
| `assignment_scheme` | see below | How `Iu` is split. Ignored under `profile: full`. |
| `extra_prompt` | string \| null | The paper's `%extra%` hook. Sweep it for cooperative/conflictual/CoT/share-all ablations. |
| `dissenter_extra_prompt` + `dissenter_fraction` | string, 0–1 | Group composition: a persona given to the first `k` agents in the recorded speaking order. |
| `aggregation` | `average` \| `majority` | Reporting only — **both** are always recorded. |
| `shuffle_facts` | bool | Per-agent seeded shuffle of `Is ∪ Iu_i` (§1.5). Turning it off is a departure from the paper. |
| `decoy` | string \| null | Null derives it per round as the modal wrong option; see §5. |

### Assignment schemes

Names come from the pipeline, not from the brief — see
[`data_provenance.md` §3](data_provenance.md#3-scheme-names-the-brief-guessed-the-pipeline-decides).

| Scheme | Availability | What it manipulates |
| --- | --- | --- |
| `bijective` | N == C only | The paper's baseline: one hidden item per agent. |
| `exact_replication` | any N ≥ C | Evidence types dealt round-robin then shuffled. At N > C a type is held by several agents, so a fact can surface from more than one source. |
| `paraphrased_replication` | prebuilt file only | Distinct wordings of one fact. **Not built yet.** |
| `factorized_evidence` | prebuilt file only | A fact split into components, recoverable only once all are pooled. **Not built yet.** |
| `padded` | any N ≥ C, mas_cc-local | Extra agents get shared information only. Isolates "more agents" from "more distributed information". |
| `decoy` | any N ≥ C, mas_cc-local | Extra agents get a shared fact restated as private — pooling noise with no new proposition. |

### `hidden_bench_vanilla` only

| Option | Notes |
| --- | --- |
| `rounds` | Communication depth `T`. `0` means no discussion. |
| `rounds_are_speaking_turns` | **Read this before quoting a `T`.** `false` (default): `T` counts full round-robin passes, so the discussion is `T x N` turns — the brief's unit, and the one that holds per-agent airtime constant as `N` varies. `true`: `T` counts speaking turns total, the paper runner's unit. They coincide only at `N = 1`. To reproduce the paper literally use `rounds: 15, rounds_are_speaking_turns: true`. |

`horizon` is **not used**: the step count is derived from `rounds` and
`n_agents`, and pricing reads `call_plan`. Configs set it to the derived value
for readability only.

Calls per episode: `N + N*T + N`, before retries.

### `hidden_bench_naming` only

| Option | Notes |
| --- | --- |
| `rounds` | Number of pair meetings. |
| `messages_per_turn` | `m` — messages each partner sends before committing. |
| `pairing` | `uniform_two_distinct` today. This is the extension point for a graph/network topology. |
| `memory_size` | `0` = unbounded. |
| `allow_relay` | Whether an agent may pass on a fact it learned from a partner. **Prompt-level, not enforced** — see §6. |
| `stop_on_consensus`, `consensus_threshold` | Early stop on standing action share. |
| `payoff.mode` | `coordination` \| `correctness` \| `coordination_plus_correctness`. |

Calls per episode: `rounds x 2 x (m + 1)`, before retries — fewer if consensus
stops it early.

**On `payoff.mode: coordination`** (the default): matching pays, being right
does not. That is deliberate. It keeps convention formation and truth-finding as
two *separable* observables; if correctness were paid, any correlation between
them in the trajectory would be an artefact of the reward, and the downstream
MI/empowerment estimators would be measuring the payoff function. All three
quantities are reported whichever mode is paying.

## 4. Metrics

Off the shared shelf (`metrics/generic.py`), unchanged, because the action
alphabet *is* the option set (`game_family: choice`):
`population_action_share_per_option`, `agent_current_action`,
`dominant_action_share`, `first_consensus_time_by_action_share`, and (naming
only) the rolling-window pair.

HiddenBench-specific (`games/hidden_bench/metrics_common.py`):

| Metric | Meaning |
| --- | --- |
| `accuracy_average` | Fraction of agents on `o*` — the paper's default rule. |
| `accuracy_majority` | 1 iff more than half are on `o*`. |
| `decoy_share` | Fraction on the decoy. |
| `unshared_disclosure_rate` | Fraction of hidden facts surfaced in conversation. **The paper's central diagnostic.** A lower bound — see §5. |
| `y_pre`, `y_post`, `improvement` | Vanilla only. `improvement = y_post - y_pre`. |
| `accuracy_first_commitment`, `accuracy_final`, `improvement` | Naming only. There is no plenary vote, so "before pooling" is per agent: the choice it made the first time it was asked. **Not the same estimator as `y_pre`**, and named differently so the two are never averaged together. |
| `disclosure_reach` | Naming only. How many distinct agents each hidden fact reached — the diffusion curve. Trivially `N` in the plenary game, which is why it only exists here. |

`gap_to_full = y_post - y_full` is **deliberately not computed by either game**:
it needs the paired `profile: full` cell, so it is resolved in analysis, where
both cells are in hand. `configs/runs/hidden_bench_grid.yaml` sweeps
`profile: [hidden, full]` precisely to produce that pair.

## 5. Two numbers to read carefully

**`unshared_disclosure_rate` is a lower bound.** Detection is normalized keyword
overlap against each hidden item (`data.py::disclosed_facts`). A faithful
paraphrase sharing few content words is missed. Read every value as "at least
this fraction was surfaced". An LLM-judge variant is the obvious follow-up and
is deliberately not v1 — it would make the benchmark's central diagnostic depend
on a second, unaudited model.

**`decoy_share` has two possible denominators.** The corpus carries no `decoy`
field. If `game.options.decoy` is set, that option is used. Otherwise the metric
falls back to the *modal wrong option in that round*, which is a within-round
statistic and **not** the same quantity as the paper's decoy. The principled
derivation — the modal wrong option under `profile: hidden, rounds: 0` — is a
property of a whole grid cell and belongs in analysis. Set `decoy` explicitly
when you want the paper's quantity.

## 6. Open decisions, for the user to sign off

### 6.1 `Control` vs. game option for message-level interventions

Reveal-All, Secretary, and Structured (Exchange-then-Decide) are **not
implemented**. The brief's §1.6 asks for a decision first, and this is it —
stated, not silently taken.

These are interventions on the *interaction*, which is conceptually the role
`control/` already plays: `runtime/loop_runtime.py` checks
`Control.override(...)` before asking the LLM, and `ForcedActionControl` uses it
to force an agent's action. The obstacle is that `Control.override` returns a
**single action value** (`str | None`) drawn from the game's action alphabet.
Reveal-All and Secretary do not force an *action* — they inject or rewrite a
*message*, and Structured changes the phase schedule itself. None of the three
fits the existing signature.

**Recommendation: option (a) — extend the `Control` protocol with a
message-level hook**, for the reason the brief gives: it makes these
interventions sweepable `grid:` axes, and therefore usable as the conditioning
variable for `mas-cc analysis empowerment`, which is the entire point of that
layer. Implementing them as game options (option (b)) is less work but leaves
them invisible to the analysis pipeline, which is where this repository is
heading.

This is not implemented because it changes a shared protocol that
`naming_convention` and the synthetic games also depend on, and that is a
decision to take deliberately rather than as a side effect of adding a game.
`control.mechanism` is accepted in the HiddenBench configs and only `none` is
wired; the orchestrator carries a comment pointing here.

### 6.2 `allow_relay` is a prompt instruction, not a constraint

May an agent quote a hidden fact it learned from a partner as if it were its
own? Default `true`, which makes `hidden_bench_naming` a genuine
information-diffusion process; `false` is a much harder ceiling in which a fact
travels at most one hop from a native holder.

**It is enforced only by telling the model.** The game cannot stop a model from
relaying, and pretending otherwise would make the `false` condition quietly
untrue. Worth deciding — and worth measuring compliance on — before any headline
number is quoted from an `allow_relay: false` run.

### 6.3 The `T` unit

See §3. The default follows the brief (round-robin passes); the paper's own
runner counts speaking turns. A `T` quoted without saying which unit it is in is
ambiguous by a factor of `N`.

## 7. Paper numbers to compare against

| Quantity | Paper | Conditions |
| --- | --- | --- |
| `Y_post` | 0.301 | Hidden profile, post-discussion, averaged over 15 models |
| `Y_full` | 0.807 | Full profile, pre-discussion |
| `Y_pre` | ~0.01–0.08 | Hidden profile, pre-discussion, 3 verification tasks, GPT-4.1 |
| `Y_full` | ~0.73 | same 3 tasks |
| `Y_post` | ~0.23 | same 3 tasks, `T = 15`, `N = 4`, 10 sessions |
| Group-size effect | `+0.348 → +0.006` | `improvement` as `N` goes 3 → 7 |
| Communication depth | peaks at `T = 15`, declines at `T = 20` | |
| Task generation yield | 57 / 200 (28.5%) | plus 3 hand-built, 5 adapted = 65 |

A large deviation from these is an implementation bug to investigate before it
is a finding. The three likeliest causes, in order: **leaked information** (run
`tests/mas_cc/test_hidden_bench_privacy.py`), **the wrong `T` unit** (§6.3), and
**vote-parsing failures being silently counted as wrong** — which cannot happen
here, because an unparseable vote raises `HiddenBenchDecisionFailed` rather than
defaulting.

## 8. Tests

```bash
conda run -n MA-CC python -m pytest tests/mas_cc/test_hidden_bench_*.py
```

| File | Covers |
| --- | --- |
| `test_hidden_bench_data.py` | §3.2 load assertions on all 65 tasks; the §4 union invariant for every (scheme, N); allocation parity with the pipeline's `N_32.json` |
| `test_hidden_bench_privacy.py` | §9.3 — no agent sees a hidden fact it does not hold, plus a self-check that the leak detector can see a real leak |
| `test_hidden_bench_prompts.py` | §9.4 — golden prompts, character for character, typos included |
| `test_hidden_bench_games.py` | §9.1 — both games end to end on the mock provider, deterministic, `call_plan` exact |
| `test_hidden_bench_grid.py` | §9.7 — a real grid through the orchestrator, read back by `analysis/reader.py::read_grid` |

Not covered, because they need a real model and real spend: §9.5 (the
reproduction against GPT-4.1) and §9.6 (the group-size trend). §7's table is
what those two would be checked against.
