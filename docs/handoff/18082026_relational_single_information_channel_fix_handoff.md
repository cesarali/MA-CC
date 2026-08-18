# Relational Reasoning — Single Information Channel Fix

Date: 2026-08-18
Game: `relational_imitation_round_feedback`
Supersedes the peer-rendering description in:
[`18082026_relational_reasoning_round_feedback_implementation_handoff.md`](18082026_relational_reasoning_round_feedback_implementation_handoff.md)

---

## 1. What was wrong

The ballot has three fields:

```json
{"vote": "B", "reason": "...", "shared_fact_id": "f2"}
```

Peers were shown both the free-form `reason` **and** the structured shared
fact. That left a second, untracked information channel: an agent could pass a
task fact — or a conclusion derived from one — inside its prose while reporting
`shared_fact_id: none`, and the fact would move without appearing in any
`K_i`.

The consequence is not cosmetic. It breaks the intended reading of

```text
K_i(t) = the exact set of task facts agent i knows
```

into a *lower bound* on what the population actually knows. Every epistemic
observable built on `K` — `mean_supporting_fact_coverage`,
`full_proof_agent_share`, `supporting_fact_reach`, peer-vs-controller
attribution — would then be unfalsifiable, because a discrepancy between
knowledge and behaviour could always be explained away as "it must have come
through the prose".

## 2. What changed

**`shared_fact_id` is now the only task-information channel between
participants.**

A visible participant renders as exactly three things — identity, vote, and the
fact it chose to expose:

```text
Agent 7
Vote: B
Evidence they are sharing:
Kavi is east of Tero.
```

With `shared_fact_id: none`, no evidence lines appear:

```text
Agent 7
Vote: B
```

The `reason` is still produced by the model, parsed, validated, applied and
**stored** — it is simply never rendered into another agent's prompt.

### The controller, consistently

The controller goes through the same renderer, so it gains no prose channel a
peer does not have. Its recommendation reaches the population as its **vote**:

```text
Agent 25
Vote: C
Evidence they are sharing:
Zora is northwest of Ralo.
```

and under `recommendation_only`, simply:

```text
Agent 25
Vote: C
```

`render_control_reason` still produces `"I recommend option C."`, now purely as
the `controller_message` label in the trajectory. It is deterministic in the
target and free of task content, so it could never have carried evidence — but
it is no longer rendered either, because the point of the fix is that *no*
participant gets a prose slot.

### One thing that had to change with it

The prompt previously told the agent:

> Your vote, your reason, and any fact you choose to share will become your
> public position and may be shown to other participants later.

That is now false, and leaving it would have been worse than the original bug:
an agent told its prose is read will *use* prose as a channel, and the run would
then be full of attempted communication that silently goes nowhere. The text now
states the actual contract:

> Other participants will see the same of you: your vote and any fact you
> choose to share, and nothing else. Your reason is your own record: it is not
> shown to anyone.

and the decision instruction makes the mechanism explicit:

> Sharing a fact is the only way to pass information to other participants.

### What did **not** change

Per the brief, none of the following was touched:

- the ballot response schema (`vote`, `reason`, `shared_fact_id`);
- `shared_fact_id` validation — the contract's task-membership check and
  `Game.validate_action`'s `shared_fact_id ∈ K_i(t)` evidence-honesty check;
- knowledge propagation — same exposures, same acquisitions, same provenance;
- controller timing, sensing, soft policy, budget, or schedule;
- q-voter slot replacement (a controlled slot still replaces one peer);
- the frozen task format or the loader;
- metrics, round records, or micro records;
- either HiddenBench game.

An agent still sees **its own** previous reason in its standing position block.
That is self-memory, not a channel — the brief only withholds *another* agent's
prose.

---

## 3. Files changed

| File | Change |
|---|---|
| `src/mas_cc/games/relational_reasoning/imitation_round_feedback/prompts.py` | `render_social_source` drops the reason line; `DECISION_BASIS_INITIAL` / `DECISION_BASIS_SOCIAL` / `DECISION_INSTRUCTION` corrected; module and constant docstrings state the single-channel rule |
| `.../runtime.py` | docstrings only — the flow diagram and `build_social_sources` now say the reason is recorded, not rendered |
| `.../game.py` | `call_plan`'s representative social source drops the `reason` key the renderer now ignores, so the token estimate matches the real prompt |
| `.../README.md` | §2, §4, §5, §7, §8, §11, §12 updated; §4 gains a "`shared_fact_id` is the only channel" subsection |
| `tests/mas_cc/test_relational_imitation_round_feedback.py` | 2 tests updated, 8 added (below) |

No config file changed. No file outside the relational game changed.

---

## 4. Tests

### Updated (2)

- `test_a_rendered_source_is_identity_vote_and_evidence_and_never_its_reason`
  (was `test_the_source_renderer_covers_both_visibilities_and_missing_evidence`)
  — now asserts the reason is dropped in **both** visibility modes while the
  record still carries it.
- `test_the_controller_appears_as_one_persistent_ordinary_participant` — the
  controller source no longer ends with a reason line.

### Added (8)

A `_ChattyBallots` provider was added whose every reason deliberately smuggles
task content (`"I know f1 and f2: Bavi is northeast of Zora, and Zora is
northwest of Ralo, so the answer must be C."`). If prose ever reached a peer,
that string would appear in someone else's prompt.

| Test | Confirms |
|---|---|
| `test_a_peers_free_form_reason_never_reaches_another_agents_prompt` | (1) reasons are produced and stored, and none is rendered into any social block; no `Reason:` line survives there |
| `test_a_social_block_shows_only_identity_vote_and_the_shared_fact` | (3) line-by-line: line 0 is the label, line 1 the vote, then either exactly the evidence header + rendered fact, or nothing — and the speaker's reason is on the record but not in the render |
| `test_the_controller_gets_no_prose_channel_a_peer_does_not_have` | (2) the controller's recommendation is recorded, its rendering is vote + evidence, and neither `"I recommend"` nor the reason text occurs in any prompt |
| `test_recommendation_only_renders_a_bare_vote_with_no_evidence_line` | (2, 3) `recommendation_only` renders `Agent 25\nVote: C` with no evidence header anywhere |
| `test_the_reason_is_still_written_to_the_trajectory_for_analysis` | (5) the reason is parsed off the response, stored on the action, on `focal_reason_after`, on the agent's `public_reason`, and survives into the serialized episode record |
| `test_an_agent_still_sees_its_own_previous_reason` | self-memory is preserved — only *other* agents' prose is withheld |
| `test_knowledge_propagation_is_unchanged_when_prose_is_withheld` | (4) a chatty and a quiet run produce identical final `K_i`, identical `new_peer_fact_ids` per event, and identical per-round coverage — the prose is inert by construction, not merely unread |
| `test_the_prompt_tells_the_agent_its_reason_is_not_shown_to_others` | the instructions do not invite a channel the runtime does not provide |

### Results

```text
tests/mas_cc/test_relational_imitation_round_feedback.py   78 passed
tests/mas_cc/test_relational_task_data.py                  18 passed
```

```bash
conda run -n MA-CC --no-capture-output python -m pytest \
  tests/mas_cc/test_relational_task_data.py \
  tests/mas_cc/test_relational_imitation_round_feedback.py -q
```

Full suite: **53 FAILED + 4 ERROR**, and the sorted failure list `diff`s
**byte-identical** to the pre-change baseline. All are pre-existing HiddenBench
failures; none is relational.

---

## 5. Smoke verification

Both shipped configs re-run clean from an empty results tree on the mock
provider. No real-model run was launched.

```bash
conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main experiment run \
  --config configs/runs/relational_reasoning/relational_imitation_round_feedback_no_control_smoke.yaml
conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main experiment run \
  --config configs/runs/relational_reasoning/relational_imitation_round_feedback_controlled_smoke.yaml
```

A manual inspection with a leaky provider (reason = `"SECRET PROSE: f1 says
Bavi is northeast of Zora."`, controller advocacy forced on, `q = 1`,
`recommendation_plus_fact` with `f2`) gives:

```text
=== SOCIAL BLOCK (controlled) ===
Agent 25
Vote: C
Evidence they are sharing:
Zora is northwest of Ralo.

=== OWN POSITION ===
Vote: C
Reason: No previous public reason.

prose in any prompt:                    False
controller recommendation in any prompt: False
reasons still logged:                    SECRET PROSE: f1 says Bavi is …
```

---

## 6. Reproducibility note

The prompt text changed, so the prompt definition hash changed:

```text
before   relational_public_ballot v1  [def:4454a5fe…]
after    relational_public_ballot v1  [def:412b6e6f…]
```

Only the two mock smoke runs existed at the old hash, and both were re-run. No
real experiment has to be re-run, and nothing else in the repository is
affected — the family and version are unchanged, so configs keep loading as
they are.

Kept at **version 1** deliberately: the family has never been used for a
recorded experiment, so bumping the version would create a second registered
prompt whose only purpose was to describe a bug. If a real run had existed, this
would have been a version bump instead.

---

## 7. What this buys, scientifically

`K_i(t)` is now an **exact** record rather than a lower bound, and that makes
the following claims checkable rather than merely plausible:

- a fact appears in `K_i` **iff** some source shown to `i` cited it — the "no
  information teleportation" invariant is now complete, not partial;
- `full_proof_agent_share > 0` on a `no_single_agent_solution` task means
  language actually moved the required evidence, and cannot be explained by
  prose leakage;
- the peer/controller split in `new_peer_fact_ids` vs `new_controller_fact_ids`
  fully partitions how information arrived, so *social diffusion* and
  *injected information* are cleanly separable;
- `reason` becomes a clean **dependent variable**: it records what an agent
  said it was doing, with no ability to affect anyone else, so prose can be
  analysed for e.g. stated-versus-revealed reasoning without being part of the
  causal path.

The corresponding cost, worth stating plainly: agents can no longer argue,
explain a chain of reasoning, or coordinate through language. Persuasion is now
"vote + at most one fact per update". That is the right trade for a first
measurement of exact information flow, and it is the thing to revisit if the
research question later becomes *how* agents argue rather than *what* moves.

---

## 8. Follow-ups this opens

Not implemented, deliberately:

- **A prose-visible arm.** The obvious follow-up experiment is a config flag
  that restores peer reasons, run as a matched comparison: same task, same
  seeds, `shared_fact_id` still recorded. The difference between the two arms
  measures exactly how much information prose carries. That would need a second
  prompt version and a `K`-lower-bound caveat on the prose arm, which is why it
  is not a quiet toggle.
- **More than one fact per ballot**, which the current single-channel rule makes
  a much more interesting knob than it used to be.
- `vote_visibility: hidden` is now close to empty (identity + evidence only) and
  should probably be dropped rather than implemented.
