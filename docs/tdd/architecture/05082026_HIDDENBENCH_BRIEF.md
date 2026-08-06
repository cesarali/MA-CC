# HiddenBench in `mas_cc` — implementation brief

**Audience.** A coding agent with full access to the `mas_cc` repository.
**Goal.** Add a `hidden_bench` game family with **two** playable games, their data,
and runnable configs.

| Game | Module | Protocol |
| --- | --- | --- |
| `hidden_bench_vanilla` | `games/hidden_bench/vanilla/` | Faithful reproduction of the paper: N agents, round-robin group discussion, pre- and post-discussion vote. |
| `hidden_bench_naming` | `games/hidden_bench/naming/` | Naming-game-style: **dyadic** interactions with private per-partner memory, on the expanded (N > 4) task data. |

The "expanded" work (more agents, redundant / factorized information assignment)
is **not** a third game — it is a *data + config axis* consumed mainly by
`hidden_bench_naming`, and available to `hidden_bench_vanilla` too.

---

## 0. Read this first: what the agent must discover, not assume

The repo already contains preprocessing work that this brief cannot see. **Before
writing any game code**, locate and read the following, and write a short
`docs/hidden_bench/data_provenance.md` recording what you found:

1. **The downloaded HiddenBench corpus.** It is somewhere under `data/`, plausibly
   `data/hidden_bench_preflight/` or a similarly named folder; the raw upstream
   release may be the Hugging Face / GitHub drop referenced in the paper, and may
   be sitting in something like a `paper_prompts/` subfolder. Find the actual path,
   the actual file format (JSON / JSONL / parquet), and the actual field names.
2. **The loader script** that fetched or normalized that corpus.
3. **The expansion script** that takes 4-agent tasks and produces >4-agent variants
   with redundant / factorized information. **The names of the expansion schemes in
   that script are authoritative** — Section 4 of this brief lists a *prior guess*
   at the taxonomy, not the truth. Reconcile them and report any mismatch instead
   of silently renaming.
4. `docs/howto/building/building_a_game.md` — the manual for the `Game`/prompt/agent
   abstractions, written against `naming_convention`. Follow its conventions exactly.
5. `docs/howto/building/running_an_experiment.md` — how a resolved config becomes a
   priced, concurrent, resumable experiment.
6. `src/mas_cc/games/naming_convention/` — the reference implementation to mirror
   (`game.py`, `prompts.py`, `metrics.py`).

If any of (1)–(3) is missing or half-finished, **stop and report** rather than
regenerating the data — the corpus and the expansion are already the user's work.

---

## 1. The paper, in the detail needed to implement it

Li, Naito & Shirado, *Systematic Failures in Collective Reasoning under Distributed
Information in Multi-Agent LLMs* (ICML 2026). HiddenBench = 65 tasks built on the
Hidden Profile paradigm (Stasser & Titus, 1985).

### 1.1 Formal model

- `N` agents, `i = 1..N`. Options `O = {o_1..o_K}`, `K >= 3`, unique correct `o* ∈ O`.
- Information `I` splits into **shared** `Is` (every agent gets it) and **unshared**
  `Iu = I \ Is`, partitioned so agent `i` gets a unique `Iu_i`, with `⋃_i Iu_i = Iu`.
- Agent knowledge: `I_i = Is ∪ Iu_i`. **Agents are never told that information is
  asymmetric.**
- `d_pre_i = f(I_i)` → discussion of `T` rounds producing messages `M` → `d_post_i = f'(I_i, M)`.
- **Hidden Profile condition holds** iff `∃i: d_pre_i ≠ o*` while pooling yields
  `f'(⋃_i I_i, M) = o*`.

Construction constraints (also the generator's design rules):

- Shared info alone points at a **decoy** (a specific wrong option).
- Shared info **+ any single** hidden item still points at the decoy.
- All hidden items pooled ⇒ the decoy is disqualified and `o*` is uniquely determined.
- **Every hidden item matters**: drop any one and `o*` is no longer uniquely identifiable.

### 1.2 Conditions

- **Hidden Profile**: agent `i` sees `Is ∪ Iu_i`.
- **Full Profile**: every agent sees `Is ∪ Iu` from the outset. This is the ceiling
  `Y_full` and must be implementable as a config flag, not a separate game.

### 1.3 Aggregation rules

- **average rule** (default): fraction of agents voting `o*`.
- **majority rule**: 1 iff more than half of agents vote `o*`.

Reference quantities: `Y_pre` (hidden, pre-discussion), `Y_post` (hidden,
post-discussion), `Y_full` (full profile, pre-discussion). Headline results:
`Y_post = 0.301` vs `Y_full = 0.807` across 15 models.

### 1.4 Task validation thresholds (reuse these as data assertions)

A task counts as a valid Hidden Profile task iff, measured **pre-discussion** and
averaged over 10 sessions:

- Full Profile per-agent accuracy **≥ 0.80** (task is individually solvable), and
- Hidden Profile per-agent accuracy **≤ 0.20** (no agent can win without pooling).

The generator passed 57 / 200 candidates (28.5%); + 3 hand-built + 5 adapted from
human studies = 65 tasks.

### 1.5 Prompts (reproduce verbatim in `vanilla/prompts.py`)

System prompt, discussion:

```
%description%

You have received the following information, notice the order of these information are
randomly shuffle, the order of facts does not indicate importance or relationship,
please reason carefully:

%information%

Keep your response concise-just one or two sentences. %extra%
```

User prompt, first speaker: `You are the first to speak.`

User prompt, subsequent speaker:

```
Previous messages from other people:
%messages%
It's your turn to speak. %extra%
```

Pre-discussion vote:

```
Please decide and provide your rationale in the following JSON format:
{
    "vote": <A string, %possible_answers%>,
    "rationale": <A string, representing your rationale>
}
```

Post-discussion vote: same, preceded by `Previous messages from other people:\n%group_discussion%`.

Two details that are easy to miss and matter for reproduction:

- **Facts are shuffled per agent**, and the prompt says so explicitly. Shuffle must be
  seeded from the episode RNG so runs are reproducible.
- `%extra%` is the hook where prompting-strategy ablations (cooperative / conflictual /
  CoT / "share all information" / adversarial-dissent persona) are injected. Make it a
  first-class, sweepable config field, not a hardcoded string.

### 1.6 Ablation axes worth supporting from day one

| Axis | Values in the paper | Where it belongs |
| --- | --- | --- |
| Communication depth `T` | 5, 10, 15, 20 (peak at 15, declines at 20) | game option |
| Group size `N` | 3, 4, 5, 6, 7 (improvement collapses `+0.348 → +0.006`) | game option + expanded data |
| Prompting strategy | cooperative↔conflictual, zero-shot CoT, informing asymmetry, share-all | `%extra%` |
| Group composition | 0–100% adversarial dissenters | per-agent `%extra%` |
| **Reveal-All** | agent appends all its info to its round-1 message | see below |
| **Secretary** | dedicated agent summarizes disclosures each round | see below |
| **Structured (Exchange-then-Decide)** | 2 exchange rounds (share 1–2 facts + one reason the front-runner is wrong), then 1 decide pass | see below |

**Architectural note worth raising with the user.** Reveal-All, Secretary and
Structured are *interventions on the interaction*, which is conceptually the same
role the existing `control/` layer plays for the empowerment analysis (`Control.override(...)`
checked by `runtime/loop_runtime.py` before asking the LLM). `ForcedActionControl`
forces an *action*; these force or inject a *message*. Two options — pick one and
justify it in the PR description:

- **(a)** Extend the `Control` protocol with a message-level hook. Wins: these
  interventions become sweepable `grid:` axes and therefore usable as the conditioning
  variable for `mas-cc analysis empowerment`, which is the whole point of that layer.
- **(b)** Implement them as game options inside `hidden_bench`. Simpler, but they stay
  invisible to the analysis pipeline.

(a) is the better fit for where this repo is going; do not implement it silently — flag it.

---

## 2. Target layout

```
src/mas_cc/games/hidden_bench/
├── __init__.py
├── data.py              # task loading, validation, information assignment
├── schemas.py           # HiddenProfileTask, AgentInfoSet (frozen dataclasses)
├── vanilla/
│   ├── game.py          # HiddenBenchVanillaGame(Game)
│   ├── prompts.py       # PromptBlocks, verbatim from §1.5
│   └── metrics.py       # to_round_view + build_metrics
└── naming/
    ├── game.py          # HiddenBenchNamingGame(Game)
    ├── prompts.py
    └── metrics.py

data/hidden_bench/        # exact path TBD by discovery (§0) — do not invent a new one
├── vanilla/              # 65 upstream tasks, normalized
└── expanded/             # N>4 variants, one file per (scheme, N)

configs/runs/
├── hidden_bench_vanilla.yaml
├── hidden_bench_naming.yaml
└── hidden_bench_grid.yaml

tests/games/hidden_bench/
docs/hidden_bench/data_provenance.md
```

Register both games in `games/registry.py`. `GameSpec.game_family = "choice"` for
both — the action alphabet is the option set, so the existing `metrics/generic.py`
shelf (`ActionSharePerOption`, `DominantValueShare`, `FirstConsensusTime`, …) applies
unchanged.

---

## 3. Data contract

### 3.1 Normalized task record

Upstream field names (from the generator's output spec) — keep them:

```python
@dataclass(frozen=True)
class HiddenProfileTask:
    name: str                          # "evacuation_west_city"
    description: str                   # scenario shown to everyone
    shared_information: tuple[str, ...]
    hidden_information: tuple[str, ...] # one item per participant in the ORIGINAL task
    possible_answers: tuple[str, ...]   # >= 3
    correct_answer: str                 # must be in possible_answers
    source: str                         # "manual" | "adapted" | "generated" | "expanded"
    n_agents_native: int                # len(hidden_information) as authored
```

### 3.2 Load-time assertions (fail loudly)

- `correct_answer in possible_answers`; `len(possible_answers) >= 3`.
- `len(hidden_information) == n_agents_native`; no duplicate hidden items.
- No hidden item is empty; no fact appears in both `shared_information` and `hidden_information`.
- If the corpus ships per-task validation stats (§1.4), assert them and record which
  tasks fail rather than dropping them silently.

---

## 4. Information assignment (the "expanded" part)

`data.py` exposes one function:

```python
def assign(task: HiddenProfileTask, n_agents: int, scheme: str, rng) -> dict[AgentId, AgentInfoSet]
```

`AgentInfoSet` = `(shared: tuple[str,...], private: tuple[str,...])`. Under
`profile: full`, `private` is the entire `Iu` for every agent, and `scheme` is ignored.

**Prior guess at the scheme taxonomy — replace with whatever the user's expansion
script actually implements (§0.3):**

| Scheme | Behaviour when `n_agents > n_agents_native` | What it manipulates |
| --- | --- | --- |
| `bijective` | Only valid when `n_agents == n_agents_native`. One item per agent. | The paper's baseline. |
| `redundant` | Each hidden item is given to `r > 1` agents. | Breaks "every hidden item matters" — a fact can now surface from several sources, so pooling gets *easier* per fact while the pool gets noisier. Expose `redundancy` as a swept parameter. |
| `factorized` | A hidden item is split into sub-statements distributed across agents; the fact is only recoverable once all its pieces are pooled. | Raises the coordination depth per fact instead of the fact count. |
| `padded` | Extra agents receive no private fact (shared only). | Group-size confound control: isolates "more agents" from "more distributed information". |
| `decoy` | Extra agents receive irrelevant or decoy-supporting facts. | Tests whether added agents actively *harm* pooling. |

Invariant every scheme must preserve and assert: **the union of all agents' private
information equals `Iu`** (modulo `factorized`, where the union of pieces must
reconstruct `Iu`). Write a property test for this.

Assignment must be seeded and logged: the realized assignment goes into the run
artifact so the post-hoc analysis can condition on *who knew what*.

---

## 5. `hidden_bench_vanilla`

Mapping onto the `Game` ABC (`games/protocols.py`). Every method below must be fully
implemented — the ABC allows no partial game.

| Method | Behaviour |
| --- | --- |
| `initialize` | Load task, build `assign(...)`, shuffle each agent's fact list (seeded), set phase to `PRE_VOTE`. |
| `select_participants` | `PRE_VOTE` / `POST_VOTE`: all `N` agents. `DISCUSS`: the single agent whose turn it is (round-robin; speaking order fixed per episode and recorded). |
| `construct_observations` | `PRE_VOTE`: description + own info. `DISCUSS`: + transcript so far. `POST_VOTE`: + full transcript. Nothing from another agent's *private state* ever enters an observation. |
| `build_decision_requests` | Compose the §1.5 prompts via `PromptBlock`/`FullPrompt`; inject `%extra%`. |
| `parse_action` | Vote phases: parse the JSON `{"vote", "rationale"}`, tolerating fenced code blocks and stray prose. Discussion: the message text is the action. |
| `validate_action` | Vote must be in `possible_answers` after normalization (case/whitespace/punctuation-insensitive; keep the raw string too). Invalid → the loop's existing retry path. |
| `apply_transition` | Append the message to the transcript, or record votes; advance phase/round. |
| `detect_termination` | After `POST_VOTE`. Discussion ends after `T` rounds (`T` counted as full round-robin passes — state this in the docstring, since the paper's `T ∈ {5,10,15,20}` is only reproducible if you match their unit). |
| `call_plan` | Exact per-episode call count for `planning/` preflight: `N` (pre-vote) `+ N*T` (discussion) `+ N` (post-vote), before retries. |

**Phase model.** `PRE_VOTE → DISCUSS × T → POST_VOTE`. Under `profile: full` with
`rounds: 0`, the same game yields `Y_full` with no discussion — that's how the ceiling
is measured, and it's why Full Profile must not be a separate game.

**Rationales are data.** Store every `rationale` string; they are the raw material for
any later embedding/InfoNCE analysis (§3.2 of the architecture overview), and the
qualitative asymmetry-signalling analysis in the paper.

---

## 6. `hidden_bench_naming`

The idea: keep the Hidden Profile *information structure*, replace the plenary
discussion with the Ashery/Baronchelli naming-game *interaction* structure.

**Interaction.** Each round, sample a pair `(i, j)` from `N` agents (sampling policy
configurable: uniform random pair, or a graph/network topology — leave a hook, since
network structure is where this repo is headed). The pair exchanges `m` messages
(config, default 1 each), then **each independently commits** to an option from
`possible_answers`.

**Memory.** Each agent keeps `AgentState` private memory: its own past partners, its
own messages and commitments, and the payoffs it received — **never another agent's
private information, and never a global transcript.** This is the key departure from
vanilla and the reason the game belongs in this repo: it makes the transcript a
*local* object, so information has to diffuse through the interaction graph rather
than being broadcast. Whether memory is full history or a bounded window is a config
option; the naming-convention game's existing memory handling is the reference.

**Payoff — both modes required (user's explicit request), selected by config:**

```yaml
game:
  options:
    payoff:
      mode: coordination            # coordination | correctness | coordination_plus_correctness
      match_reward: 1.0
      mismatch_penalty: -1.0
      correctness_bonus: 0.5        # used only by coordination_plus_correctness
```

- `coordination` — pure Ashery-style: `+match_reward` iff both agents committed to the
  same option, `mismatch_penalty` otherwise. **Correctness is not paid, only measured.**
  This keeps convention formation and truth-finding as two separable observables, which
  is what you want before feeding anything into the MI/empowerment estimators.
- `correctness` — payoff on agreement with `o*` only; closer to HiddenBench proper.
- `coordination_plus_correctness` — matching pays, matching on `o*` pays
  `match_reward + correctness_bonus`.

Report all three quantities as metrics **regardless of which mode is paying**, so runs
across modes stay comparable.

**Termination.** `detect_termination` fires on whichever comes first: max rounds, or a
consensus criterion reusing `FirstConsensusTime`'s threshold semantics. Both configurable.

**`call_plan`.** `rounds × 2 × (m + 1)` calls per episode (message turns + commitment),
before retries.

**Open design question to surface, not to decide alone:** whether an agent may *quote*
a hidden fact it learned from a partner as if it were its own (free propagation), or
whether it may only pass on facts it holds natively. The former makes the game a genuine
information-diffusion process; the latter is a much harder ceiling. Default to the
former, expose it as `allow_relay: true`, and say so in the docstring.

---

## 7. Metrics

Both games, via `to_round_view` adapters into the existing shelf:

- `population_action_share_per_option`, `agent_current_action`, `dominant_action_share`,
  `first_consensus_time_by_action_share` — reuse unchanged.

HiddenBench-specific, in each game's `metrics.py`:

- `accuracy_average` — fraction of agents on `o*` (the paper's default).
- `accuracy_majority` — 1 iff `> N/2` agents on `o*`.
- `y_pre`, `y_post`, `improvement = y_post - y_pre`, `gap_to_full = y_post - y_full`
  (the last as an episode-level `FinalMetric`; `y_full` comes from the paired
  `profile: full` cell of the grid, resolved in analysis, not inside the game).
- `decoy_share` — fraction on the decoy option. Requires a `decoy` field; if the corpus
  doesn't carry one, derive it as the modal wrong option under `profile: hidden,
  rounds: 0` and record how it was derived.
- `unshared_disclosure_rate` — fraction of hidden facts that were surfaced in the
  transcript by round `t`. **This is the paper's central diagnostic** (agents integrate
  fine but fail to *surface*). Start with normalized substring / keyword matching against
  each hidden item; an LLM-judge variant is a follow-up, not v1. Be explicit in the
  docstring that this is a lower bound — paraphrased disclosures will be missed.
- Naming only: `disclosure_reach` — number of *distinct agents* that have been exposed
  to each hidden fact, i.e. how far each fact diffused through the interaction graph.

---

## 8. Configs

`configs/runs/hidden_bench_vanilla.yaml`, mirroring the existing run-config shape
(don't invent a new schema — copy `configs/runs/` conventions):

```yaml
game:
  type: hidden_bench_vanilla
  options:
    task_set: vanilla          # vanilla | expanded
    task_id: evacuation_west_city   # or task_ids: [...] / all
    n_agents: 4
    profile: hidden            # hidden | full
    assignment_scheme: bijective
    rounds: 10                 # T, full round-robin passes
    extra_prompt: null         # the %extra% hook
    aggregation: average       # average | majority (reporting only)
    shuffle_facts: true
    seed: 0
control:
  mechanism: none              # or reveal_all / secretary / structured, if §1.6(a) is taken
```

`hidden_bench_naming.yaml` adds `pairing`, `messages_per_turn`, `payoff`, `memory`,
`allow_relay`, and defaults to `task_set: expanded` with `n_agents: 5`.

`hidden_bench_grid.yaml` — a `grid:` sweep over the axes that matter first:
`profile × n_agents × assignment_scheme × rounds`. Confirm the cartesian product,
shared concurrency pool and budget preflight behave as `experiments/orchestrator.py`
expects, and that `mas-cc experiment preflight` refuses to launch what it can't price.

---

## 9. Acceptance checks

Ordered — do not proceed past a failing one.

1. **Unit, mock provider.** Both games run end to end against the mock LLM provider.
   Deterministic under a fixed seed. `call_plan` matches the observed call count exactly.
2. **Data properties.** All §3.2 assertions pass on the full corpus; the §4 union
   invariant holds for every `(scheme, n_agents)` pair; every scheme is exercised.
3. **Privacy.** A test asserting no agent's observation ever contains another agent's
   private fact under `profile: hidden` — except via that agent's own message text.
   This is the single most important correctness property of the whole benchmark;
   if it leaks, every number is meaningless.
4. **Prompt fidelity.** Golden-file test: rendered prompts match §1.5 character for
   character (including the paper's own typos — `"randomly shuffle"`, `"concise-just"`).
5. **Reproduction, small.** On the 3 verification tasks with GPT-4.1, 10 sessions,
   `T = 15`, `N = 4`, expect roughly: `Y_pre ≈ 0.01–0.08`, `Y_full ≈ 0.73`,
   `Y_post ≈ 0.23`. Wide tolerance, and treat a large deviation as an implementation
   bug to investigate (most likely: leaked information, wrong `T` unit, or a
   vote-parsing failure being silently counted as wrong).
6. **Group-size trend.** On expanded data, improvement `Y_post − Y_pre` should decay
   with `N`. Directional check only — the paper's `+0.348 → +0.006` used different tasks.
7. **Analysis wiring.** A completed grid dir is readable by `analysis/reader.py::read_grid`
   and `mas-cc analysis empowerment --grid-dir` runs without modification.

---

## 10. Deliverables

- The two games, registered, with docstrings that cite the paper section they implement.
- Normalized data under `data/hidden_bench/` (path per §0), plus `data_provenance.md`.
- The three configs.
- Tests per §9.
- `docs/hidden_bench/README.md`: how to run each game, what each config axis means,
  and a table of the paper numbers to compare against.
- A short note on the §1.6 `Control`-vs-game-option decision, for the user to sign off.

## 11. Explicitly out of scope

- Regenerating tasks with the GPT-4.1 generation pipeline (the corpus is already downloaded).
- The embedding / InfoNCE estimator path.
- Human-subject comparison.
