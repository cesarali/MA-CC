# ASTRA task-003 no-controller baseline (q12_deepinfra_rho3)

Prepared only; no simulation or SLURM job has been launched.

Matched to the existing false-control and truth-control studies for this
provider: same task-003, model, prompts, population of 24, 30 rounds, 30
repetitions, root seed 20260907, and paired initialization artifacts.
Persistence values are 0.70, 0.85, and 1.00: 3 cells and 90 episodes.
The social group size is 12.

The control mechanism is `none`, with empty control options. There is no
controller sensing, intervention, request, directive, or report generation.
Participant communication and participant requests remain enabled.
There is no budget sweep; metadata records intervention budget zero.
This one autonomous arm is the baseline for both target-control arms.

The analysis recipe retains the matched weighted summaries, epistemic metrics,
calibration variants, 1000 bootstrap resamples, 95% confidence, and seed 1.
No-controller data cannot identify a randomized controller treatment effect.
Causal susceptibility and controller efficiencies must remain absent,
unsupported, or undefined as appropriate, never presented as zero effects.
Model predictions remain disabled. Target-relative descriptive outputs use
the runtime's baseline reference and must not be confused with a false target.

Use the existing generic study workflow and launcher; no custom SLURM job or
replacement estimator was added. Results are configured under
`/work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/astra_task003_no_control_q12_deepinfra_rho3`.

Preflight (2026-09-14) permitted this study: 64,800 nominal, 76,320 expected,
and 324,000 conservative provider calls. These are planned calls, not calls
made during preflight. The ordinary preflight report's `NOT_REQUESTED`
scientific-contract label means no additional named design contract was
requested; experiment launch checks passed.
Report:
`/work/ojedamarin/Projects/LanguageGames/MA-CC/results/preflight/astra_task003_no_control_q12_deepinfra_rho3_20260914/report.md`.

Submission requires separate authorization. Recheck pricing, provider load
shared with any other running studies, initialization availability, and the
execution plan before launch.
