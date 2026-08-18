# HiddenBench Round Feedback — Controller Shared-Fact Evidence

Date: 2026-08-18
Game: `hidden_bench_imitation_round_feedback`
Commit: `03b4224` ("New strong steer experiment")
Option: `control.options.evidence_mode`

Related, but a **different game and a different design** — see §8:
[`18082026_relational_reasoning_round_feedback_implementation_handoff.md`](18082026_relational_reasoning_round_feedback_implementation_handoff.md)

---

## 1. What this adds

The controller's public reason could previously say only one thing, a fixed
line with no factual content:

```text
Agent 25
Vote: West City
Reason: Based on the discussion so far, West City still looks like the
strongest option to me and deserves more weight.
```

`evidence_mode: shared_fact` lets it support that advocacy with exactly one
piece of task evidence:

```text
Agent 25
Vote: West City
Reason: Based on the discussion so far, West City still looks like the
strongest option to me and deserves more weight. Relevant information: The
local government announced that hotels in West City are prepared to
accommodate evacuees. These hotels are fully stocked with food and medical
supplies.
```

The point is to make **bare advocacy vs. evidence-supported advocacy** a
measurable axis: same target, same budget, same sensing, same slot — only the
justification differs.

### The constraint that makes it sound

The cited fact comes from the task's **shared** set `Is` and from nowhere else.
`Is` is the block every agent is already shown in its own prompt. So the mode
changes how the control signal is *justified*, **not how much information the
population has**. It is a presentation axis, not an information axis.

---

## 2. Configuration

`control.options.evidence_mode` — a flat scalar alongside `target`, `beta`,
`intervention_budget`. Deliberately not the nested `controller_evidence: {mode:
…}` form the original request sketched: there are no nested control options
anywhere in this repository, and matching the surrounding style mattered more
than matching the sketch.

```yaml
control:
  mechanism: round_soft_target_budgeted
  options:
    target: correct
    sensor_sample_size: 12
    policy: soft_target
    threshold: 0.5
    beta: 4.0
    intervention_budget: 6
    template_version: 3
    evidence_mode: none          # bare advocacy — identical to omitting the key
    # evidence_mode: shared_fact # advocacy + one true shared fact
```

| Value | Controller's public reason |
| --- | --- |
| `none` (default) | `Based on the discussion so far, Z still looks like the strongest option to me and deserves more weight.` |
| `shared_fact` | …the same line, then `Relevant information: <one shared fact>` |

As the grid axis this exists for:

```yaml
grid:
  control.options.evidence_mode: [none, shared_fact]
```

An unrecognised value is refused at control creation:
`control.options.evidence_mode: must be one of ['none', 'shared_fact']`.

---

## 3. Fact selection

**No LLM call, no semantic-selection mechanism, nothing fabricated.**

Selection reuses machinery the parent game already had —
`games/hidden_bench/imitation/controller.py::_preferred_facts` — whose ordering
is: shared facts that name the target first, then those of at most two
sentences, then the whole shared pool. That is the only fact→option mapping the
corpus actually carries (the option label appearing inside the fact text), so on
a task whose facts never name an option this degrades to a uniform seeded draw
rather than to a semantic guess.

On the evacuation task the mapping is clean and effectively deterministic:

```text
West City  -> [fact 0]        # names West City
East Town  -> [fact 1]        # names East Town
North Hill -> [fact 2, 3]     # two candidates -> seeded draw
```

Facts containing an apparatus word (`controller`, `external`, `experiment`,
`simulation`, i.e. `FORBIDDEN_MESSAGE_TERMS`) are dropped. No canonical corpus
fact contains one today — the only near-hit is `system`, in nine facts, which is
not a forbidden term — but the existing "no prompt ever identifies a source as
control" invariant would otherwise silently become a live risk the first time
someone adds a task. If filtering leaves nothing, the round cites no fact and
logs `null`; it never raises mid-episode and kills a grid.

### Two cadence decisions

**One fact per controller decision, not per actuated slot.** The draw happens
once per round, from `Seed(...).derive(f"round-feedback-controller-evidence:{round_index}")`,
so all `b` controlled positions of a round show the population the same message.
Drawing per slot would make `U_t → M_t` a second random channel on top of
`Y_t → U_t` — which is the exact reason §5.6 of the game doc rejects
`template_version: 2`, and the actuation estimate could no longer say which
channel the population responded to.

**Reasoning mode only.** Classical dynamics renders no controller text at all —
a controlled slot is a term in the strict-unanimity kernel, not a message — so
`controller_evidence_fact` stays `null` there whatever the mode says. Selecting
in classical mode would log evidence no agent was ever shown.

---

## 4. Files changed

All under `src/mas_cc/games/hidden_bench/imitation_round_feedback/`:

| File | Change |
| --- | --- |
| `controller.py` | `evidence_mode` field on `RoundSoftTargetBudgetedControl` + validation in `_extra_from_options`; new `select_shared_evidence_fact(state, target, rng)`; `CONTROL_EVIDENCE_MODES` / `EVIDENCE_NONE` / `EVIDENCE_SHARED_FACT` |
| `prompts.py` | `render_control_reason(target, *, evidence_fact=None)` — new keyword-only arg, default preserves the old string exactly; `CONTROL_EVIDENCE_PREFIX = "Relevant information:"` |
| `runtime.py` | reads the mode off the control; per-round selection; `controller_evidence_fact` threaded through `build_social_sources`; three new round-record fields |

Plus:

- `docs/documentation/games/imitation_feedback/imitation_round_feedback.md` —
  new §5.7, renumbered old §5.7→§5.8 and §5.8→§5.9, round-record field list,
  error table row.
- `tests/mas_cc/test_hidden_bench_round_feedback_public_ballot.py` — 5 new
  tests, `_BudgetedControl` gained an optional `evidence_mode` arg.

Nothing outside this package was touched. The q-voter/controller timing, the
control budget, the sensing mechanism, and the public ballot are untouched.

### Where the shared facts are read from

`_shared_information(state)` reads `agent.attributes["shared_information"]` —
**not** `state.data["task"]`, which does not carry the shared set (it carries
`hidden_information`, `possible_answers`, `correct_answer`). The unshared pool
`Iu` is not reachable from the selector at all.

Note also that `game.py::_publish_focal_ballot` credits a focal agent with facts
it read by matching against `hidden_information`. A cited *shared* fact never
matches, so `known_facts` / `disclosure_reach` stay uncontaminated with no extra
code.

---

## 5. Logging

Three fields on every round record (`round_trajectory.jsonl`, the compact
scientific channel kept under every artifact profile):

```text
controller_evidence_mode        'none' | 'shared_fact'   (as configured)
controller_evidence_fact        the exact text, or null
controller_evidence_fact_index  its position in Is, or null
```

A real record from a `shared_fact` smoke run:

```text
controller_action:              'ADVOCATE_Z'
controller_target:              'West City'
controller_evidence_mode:       'shared_fact'
controller_evidence_fact:       'The local government announced that hotels in
                                 West City are prepared to accommodate evacuees.
                                 These hotels are fully stocked with food and
                                 medical supplies.'
controller_evidence_fact_index: 0
```

Microscopic rows in `trajectory.jsonl` already carried `social_sources` with the
full rendered `reason`, so the fact is recoverable per update without adding
per-update columns.

---

## 6. Backward compatibility

Structural, not merely asserted:

- The field defaults to `none`; the runtime reads it with
  `getattr(resolved_control, "evidence_mode", "none")`, so a `Control` that
  predates the option — or a test double — renders the historical advocacy.
- `render_control_reason`'s new argument is keyword-only with a `None` default,
  and an empty/whitespace fact is treated as absent.
- Every pre-existing test in the file exercises the default path, because the
  test harness's `_BudgetedControl` does not set the option unless asked.

**The prompt definition hash is unchanged.** Verified directly:

```text
evidence_mode: none         930f3ca463fc90b894a11f20298aa5b7eb756d7734998433962829372ab92f1c
evidence_mode: shared_fact  930f3ca463fc90b894a11f20298aa5b7eb756d7734998433962829372ab92f1c
rendered prompts differ:    True
```

The fact enters as a *bound value* of an existing block, not as a change to the
prompt definition, so `definition_hash` (which fingerprints
`definition_fingerprint_dict()`, excluding bindings) is untouched while
`instance_hash` moves. **No recorded experiment needs re-running**, and the
family/version stay at `hidden_bench_public_ballot` v1. Contrast the sibling
relational fix, where prompt *text* changed and the hash moved.

---

## 7. Tests

Five added to `tests/mas_cc/test_hidden_bench_round_feedback_public_ballot.py`,
all mock-provider, no LLM calls:

| Test | Asserts |
| --- | --- |
| `test_evidence_mode_defaults_to_none_and_is_validated` | absent key → `none`; `shared_fact` accepted; unknown value raises `ConfigurationError` |
| `test_evidence_mode_none_renders_the_historical_fact_free_advocacy` | explicit `none` and an absent key produce byte-identical reasons; round records log `none` / `null` |
| `test_shared_fact_mode_appends_one_true_shared_fact_that_names_the_target` | reason starts with the bare line; cited text is verbatim in `Is`; it names the target; `fact_index` indexes it; the fact actually reaches a prompt |
| `test_shared_fact_mode_never_cites_unshared_information` | no `hidden_information` item appears in any control reason; no apparatus word leaks into any prompt |
| `test_the_cited_fact_is_one_per_round_and_replays_from_the_seed` | two runs cite the same sequence; at most one distinct control reason per round |

The determinism test targets `North Hill` on purpose — it is the one option with
two on-target shared facts, so the draw is a real draw rather than a forced
single candidate.

**Mutation-checked.** Forcing `controller_evidence_fact=None` at the
`build_social_sources` call site fails 2 of the 5. They bite.

### Regression check

```bash
conda run -n MA-CC python -m pytest tests/mas_cc/test_hidden_bench_round_feedback_public_ballot.py -q
# 34 passed
```

Against the whole suite, the failure *sets* before and after are identical —
compare sorted `FAILED`/`ERROR` lists across a `git stash`, never the count:

```bash
pytest tests/ -q 2>&1 | grep -E '^(FAILED|ERROR tests)' | sort > after.txt
git stash && pytest tests/ -q 2>&1 | grep -E '^(FAILED|ERROR tests)' | sort > before.txt; git stash pop
diff before.txt after.txt    # empty
```

---

## 8. How this differs from the relational-reasoning controller

The sibling game shipped its own controller-evidence feature the same day. They
solve the same brief differently, and **the divergence is deliberate but
unreviewed** — worth a decision if the two are ever meant to be compared
directly.

| | `hidden_bench_imitation_round_feedback` (this) | `relational_imitation_round_feedback` |
| --- | --- | --- |
| Option | `evidence_mode: none \| shared_fact` | `message_mode: recommendation_only \| recommendation_plus_fact` |
| Fact pool | task's shared set `Is` | the frozen task's fact list |
| Selection | relevance heuristic + seeded draw | explicit `controller_fact_id`, or `controller_fact_selector: supporting` |
| Resolved | once per **round** | once per **episode** |
| Rendering | appended prose, `Relevant information: <fact>` | structured `Evidence they are sharing:` field, the same renderer peers use |
| Fact enters `K_i` | n/a — shared facts are not tracked knowledge | yes, recorded as `controller_fact_id` |

The rendering difference is the substantive one. The relational game made the
controller's evidence travel the *same structured channel a peer uses*, so
controlled slots are not identifiable by shape. Here the controller's citation
is prose appended to its reason — which is fine, because in this game peers also
write free prose reasons, so a controller reason containing a sentence of shared
evidence is exactly the shape of an ordinary participant's reason. The
indistinguishability argument holds in both, by opposite routes.

---

## 9. Known limitations and follow-ups

1. **`shared_fact` says nothing about the fact's *direction*.** `_preferred_facts`
   selects facts that *mention* the target, not facts that *support* it. On the
   evacuation task the fact naming West City happens to favour it, but nothing
   enforces that, and on a task engineered around a shared-information decoy the
   controller could cite an on-target fact that reads as evidence against its own
   target. If the experiment needs "supporting evidence" rather than "relevant
   evidence", that needs a corpus annotation, not a code change here — the
   relational game solved it with an explicit `supporting` selector.

2. **A pre-existing doc inconsistency, flagged not fixed.** §5.6 of the game doc
   quotes the version-3 message as *"Weighing up the discussion so far, **Z**…"*
   (the parent's `fixed_advocacy_message`), but the reasoning path actually
   renders `render_control_reason`'s *"Based on the discussion so far, Z…"*. The
   new §5.7 table uses the real text, so the two sections now visibly disagree.
   `controller.py`'s module docstring already says the parent's message "is
   **not** what a reasoning-mode agent sees"; §5.6 just never made it explicit.
   One-line fix, outside this change's scope.

3. **The suite is red on `main`, independently of this work.** 53 `FAILED` + 4
   `ERROR`, of which 51 are one root cause: run configs the tests reference no
   longer exist — `configs/runs/hidden_bench/hidden_bench_imitation_classical.yaml`
   (×20), `..._reasoning_mock.yaml` (×10), `configs/runs/hidden_bench_vanilla.yaml`
   (×9), `hidden_bench_naming.yaml` (×6), `hidden_bench_grid.yaml` (×4),
   `..._first_control_grid.yaml` (×2). Same class as the 2026-08-06 "Reorder
   Configs" breakage, new instance. Worth a separate pass.

4. **Two `..._round_feedback.py::test_every_round_feedback_run_has_live_console_and_completed_cell_reporting`
   failures** are a smaller, different thing: `gpt-oss-steer-test.yaml` and
   `..._grid_D_no_control.yaml` lack the `configured analysis ready` line the
   test requires in their header comments.

5. **No experiment has been run with this.** The feature is implemented, tested
   against mocks, and documented; nothing has been spent on a live grid.

---

## 10. Selection audit over the steerability tasks (2026-08-18)

Run before launching
`configs/runs/imitation_round_feedback/gpt-oss-steer-test-shared-evidence.yaml`,
by initializing each task exactly as the run does and calling the real
`select_shared_evidence_fact`. `†` marks the task's correct answer.

**Headline: the two target arms do not receive evidence of equal quality.**
`_preferred_facts` selects on *mention*, and a hidden profile's shared block is
built to favour a decoy — so the `correct` arm often cites evidence against
itself while `random_incorrect` usually gets a genuinely supporting fact.

| Task | Target | Candidates | Selected fact (abbrev.) | Reads as |
| --- | --- | --- | --- | --- |
| `evacuation_north_hill` | West City | `[0]` | hotels prepared, *fully stocked with food and medical supplies* | **supports** |
| | East Town | `[1]` | mayor offered accommodation, volunteers available | **supports** |
| | North Hill † | `[2, 3]` | `[2]` two-week supply + gym space, *but lacks privacy*; `[3]` mudslide, *driveway open*, trails blocked | supports-with-caveat / **mixed** |
| `Laboratory Theft Deduction` | Lab Alpha | `[0, 2]` | *only Lab Alpha* recorded as having the code machine; Alpha has prior security violations | **supports** (incriminates Alpha) |
| | Lab Beta | `[0, 1, 2]` | whole pool — no shared fact names Beta | **⅔ opposes** |
| | Lab Gamma † | `[0, 1, 2]` | whole pool — no shared fact names Gamma | **⅔ opposes** (cites Alpha-incriminating evidence) |
| `datacenter_emergency_migration` | Alpha | `[1]` | on-call engineers present, no scheduled maintenance | **supports** |
| | Beta | `[0]` | fully online, just upgraded, low utilization | **supports** (the decoy) |
| | Gamma † | `[2]` | *minor flood warning*, but never significant weather downtime | **mixed**, leads negative |
| `emergency_supply_drop` | Warehouse A | `[1]` | A and C have paved landing zones, *but only B is adjacent to an emergency response center* | **mentions**, punchline favours B |
| | Warehouse B | `[0, 1, 2]` | fastest from airstrip / only one adjacent to response center / chemical plant no issues yet | **supports** (all three) |
| | Warehouse C † | `[1]` | same sentence as Warehouse A's | **opposes** — argues for B |
| `choosing_base_camp` | Camp Summit | `[0, 1]` | stable weather 72h, *seems safest*; extra cold-weather gear delivered | **supports** (the decoy) |
| | Camp Pinecone † | `[2]` | *reports last year of bears near the river* | **opposes** — the only fact naming it is negative |
| | Camp Meadow | `[3]` | high winds expected tonight | **opposes** |

Scoring the five `correct`-arm rows: **none** gets a cleanly supporting fact;
two are mixed, three effectively argue against the target. Scoring the ten
wrong-option rows: most get clean support, which is what `random_incorrect`
will usually draw.

### What this means for the experiment

This is a property of the corpus, not a defect in the selector — a hidden
profile is *defined* by shared information pointing the wrong way. But it makes
one comparison unsafe and one comparison fine:

- **Unsafe:** reading a `correct` vs `random_incorrect` gap *within* the
  shared-evidence run as a steerability result. It confounds "citing evidence
  helps" with "shared evidence favours wrong answers".
- **Fine:** comparing each cell against its own twin in `gpt-oss-steer-test.yaml`
  — same task, same target, same `b` — which is what the matched pair is for.

If a clean "supporting evidence" arm is ever wanted, it needs a corpus
annotation (fact → option it supports), not a change to this selector. The
relational game solved the same problem with an explicit
`controller_fact_selector: supporting`, and that is the shape the fix would
take here — see §8.

### Reproducing the audit

```bash
conda run -n MA-CC --no-capture-output python - <<'EOF'
from dataclasses import replace
from mas_cc.config.loader import load_run_config_or_grid
from mas_cc.games import create_game
from mas_cc.games.hidden_bench.imitation.controller import _preferred_facts, _shared_information

spec = load_run_config_or_grid(
    "configs/runs/imitation_round_feedback/gpt-oss-steer-test-shared-evidence.yaml",
    environment={},
)
tasks = next(a.values for a in spec.axes if a.path == "game.options.task_id")
for name in tasks:
    o = dict(spec.base.game.options); o["task_id"] = name
    cfg = replace(spec.base, game=replace(spec.base.game, options=o))
    state = create_game(cfg.game).initialize(cfg.game, cfg.execution.seed)
    shared = _shared_information(state)
    print(f"\n### {name}  correct={state.correct_answer!r}")
    for i, f in enumerate(shared):
        print(f"  [{i}] {f}")
    for target in state.possible_answers:
        print(f"  -> {target!r}: {[i for i, _ in _preferred_facts(shared, target)]}")
EOF
```

---

## 11. Where to read next

- `docs/documentation/games/imitation_feedback/imitation_round_feedback.md` §5.7
  — the user-facing reference for the option.
- `src/mas_cc/games/hidden_bench/imitation_round_feedback/controller.py`
  — `select_shared_evidence_fact` and its docstring carry the privacy argument.
- `src/mas_cc/games/hidden_bench/imitation/controller.py::_preferred_facts`
  — the relevance ordering, shared with the parent game's `template_version: 2`.
