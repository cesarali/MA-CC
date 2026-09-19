# Calibrating the semantic attribution against human labels

`mas_cc.analysis.semantic_attribution` asks System One four questions about
every posted board message (does it cite the controller, which allocation it
argues for, how hard it pushes toward the controller's target, does it state
evidence). Nothing about those answers is validated yet: the questions were
written once, the model answered, and the per-cell means went into a table.
Before any of it is read as a measurement, the answers have to be scored
against people reading the same messages. This is the protocol; the two
scripts under `scripts/Cygnus/analysis/` do the mechanical parts and make no
network calls.

## Protocol

1. **Sample.** From the judged table of one study (`semantic_attribution.parquet`)
   draw 200 messages, stratified by cell, with a fixed seed:
   ```bash
   $PY scripts/Cygnus/analysis/semantic_label_sheet.py --judged <run>/semantic_attribution.parquet \
       --sample 200 --seed 1 --output labels-A.csv
   cp labels-A.csv labels-B.csv
   ```
   Both labellers get the same messages in the same order. The sheet shows the
   message, the possible answers, the controller target, whether the agent
   could see the controller's post, and the agent's vote before and after
   posting: exactly the state the model sees, nothing more.
2. **Label independently.** Fill the four `human_*` columns; do not look at the
   model's answers (they are not on the sheet). Vocabularies:

   | column | values |
   |---|---|
   | `human_cites_controller` | `yes` / `no` |
   | `human_stance` | one of the message's possible answers, or `none` |
   | `human_pressure` | `0` no push · `1` weak · `2` moderate · `3` strong (the `PRESSURE_LEVELS` texts) |
   | `human_provides_evidence` | `yes` / `no` |

   Put a name or initials in `labeller`; use `notes` for anything ambiguous.
   Expect 200 messages to take about two hours per person.
3. **Score.**
   ```bash
   $PY scripts/Cygnus/analysis/semantic_agreement.py --judged <run>/semantic_attribution.parquet \
       --sheet labels-A.csv --sheet labels-B.csv --output agreement.json --rows agreement-rows.csv
   ```
   `agreement.json` reports, per question, model-vs-human agreement and, with
   two sheets, human-vs-human agreement on the shared messages.

## Reading the result

* **The human-human kappa is the ceiling.** A model-human kappa close to it
  means the question is answered as well as people answer it; a model-human
  kappa far below it means the question or its criteria need rewording.
  Human-human kappa below about 0.6 means the question itself is not well
  posed, and rewording comes before anything else.
* **Noul questions have no fixed threshold.** The model gives a value in
  [0, 1]; the report sweeps thresholds 0.05 … 0.95 and marks the
  kappa-maximising one. That threshold becomes the study's convention for
  turning the value into a yes/no share; the per-cell *means* in
  `semantic_attribution_summary` do not depend on it.
* **Pressure is ordinal.** Read the linear-weighted kappa and the within-one
  accuracy, not exact accuracy; adjacent levels are expected to blur.
* **Stance confusion.** Look at the `none` row: the model calling a
  question-only message an argument for the target is the failure that would
  inflate `stance_matches_target`.

## What to change when agreement is poor

Question texts and criteria live in `semantic_attribution.message_questions`
under `QUESTION_VERSION`. Bump the version when changing them, re-run the
attribution (cache keys include the questions, so nothing stale is reused),
and re-score against the same sheets; the sheets do not depend on the
question wording. Do not tune criteria to the 200 labelled messages and then
report agreement on the same 200: hold a second sample for the final number.

## Known constraint

System One's answers depend on how many messages are sent per request:
measured 2026-09-19, batching five messages per call agreed with single-message
calls on only 82.7 % of stance labels and moved the mean pressure from 0.45 to
0.98. The attribution therefore runs one message per request by default, and
any calibration result applies to that setting only.
