# Integration prompt for the Santa Fe simulation agent

You have access to the Santa Fe repository and simulation outputs. Integrate the supplied `santa_fe_theory` package as the **reduced Langevin prediction layer** alongside the microscopic Santa Fe v3 runs. Produce matched simulator-versus-theory plots and quantitative discrepancies. Do not rerun the old simplified figure script or describe the three-variable runner as full heterogeneous mean-field theory.

The package is standalone and already implements parameterized kernels, the hybrid night/day/controller clock, two participant-sampling laws, three covariance choices, forced next-day branches, CSV exports, and comparison plotting. Read its README and run its tests before integration. Use the existing `santa_fe_game_mechanics.md` and v3 validator to ground all adapter logic. No LLM calls are needed.

## 1. Add a small adapter, not another simulator

Inspect current `src/santa_fe/v3_game.py`, state records, resolved configs and `src/santa_fe/v3_validation.py`. Reuse existing loading, snapshot projection, seed and output machinery. Do not guess the meaning of a field because its name resembles the paper's notation.

The adapter should export, per physical cell:

1. `config.json` following `examples/config.json`: actual N/F+/F-/q, actual integer b and q_c, rho, the two response coefficients, controller target, and policy parameters. Copy resolved values after simulator rounding/capping; do not repeat rounding from budget_fraction/sensing_fraction. Require model_version v3.
2. `initials.json`: the actual saved initial population projection and the actual front page for each independent initial block. Include x, kappa_plus, kappa_minus and four message counts. Initial vote/evidence correlation and identity structure are discarded by the reduced model, but its starting means/board must still match exactly.
3. A canonical simulator trajectory CSV and comparison manifest described in README. Include independent block IDs and downstream replica IDs with a complete round panel. Keep truth share, target share and peer-board target share separate.
4. Pre-action snapshots for branch tests: end-of-day population plus the closed N-message peer board **before** controller injection, not the combined next front page.

The `project_snapshot` utility accepts canonical agent/message dictionaries. Translate actual repository records first. Its result includes parameter counts N/F+ /F-; verify these against the cell manifest, then use only state/board fields in the snapshot schema. It does not parse your repository's internals automatically.

Keep initial fact redundancy and any task-generation settings in provenance. The runner never reconstructs the initial state from these settings. Every initial state in one batch must share the same round origin. Use separate batch directories for different physical parameter cells or continuation origins.

## 2. Execute three clearly labeled reference variants

Use the same actual initial snapshots and physical parameters for all variants:

- `paper_reduced`: with_replacement + paper_diagonal.
- `sampling_matched_reduced`: without_replacement + paper_diagonal.
- `sampling_clock_corrected_reduced`: without_replacement + fixed_clock.

Optionally add full_poisson under the same sampling rule to isolate centering from off-diagonal covariance. Names are suggested output labels, not existing repository config switches. Read the CLI's supported values. The default runner variant is the third, so set choices explicitly in every saved config.

These comparisons distinguish the paper's original sampling/noise conventions from simulator-aligned category sampling and covariance corrections. Even the most corrected variant retains independent-binomial epistemic states, old-vote independence, Gaussian aggregate thinning, exchangeable identities, and independent board generation. Do not call it an exact simulator surrogate.

For q distinct messages, finite-board category probabilities are hypergeometric; within a category the runner still averages iid uniform fact identities. Actual duplicate IDs and holder–board overlaps remain an approximation. Preserve this caveat in captions and manifests.

## 3. Run a small verified pilot first

Start with a few already simulated balanced cells, such as rho={0.4,0.7,0.95}, b={0,6,24}, N=24, q=3 and actual q_c=12 if these cells exist. Otherwise use available cells and state their exact parameters. Do not launch the full grid until adapter/time alignment checks pass.

Export several actual initial blocks and initially use a modest 16–32 theory replicas per block. Refine Monte Carlo counts based on observed uncertainty. Run substeps 24 and 48 at selected cases; add 96 only to resolve meaningful step sensitivity. Inspect projection diagnostics and retained-state scope before interpreting discrepancies. Retain independent theory seeds; simulator seeds are identifiers of shared initial conditions, not a promise of matched Langevin noise.

Use one output directory per cell/variant. The package is a serial reference implementation. You can distribute independent cells or initial-block shards through the repository's existing job system. If sharding initial blocks, remap seeds deterministically so shards do not accidentally reuse identical streams. Preserve global block IDs and merge raw rows before recomputing summaries; do not average reported SEs.

The finite-sample enumeration has a resource guard. If F or q is large, estimate memory/time before increasing it. Optimize/cache only with regression checks against existing kernels. Do not silently switch to a mean-input response to make a job faster.

## 4. Generate the actual overlays

Use `python -m santa_fe_theory run ...`, then `compare ...` with adapted **real simulator outputs**. The supplied illustrative examples and interface-test data are not validation data.

Time alignment is mandatory:

- Round 0 is the supplied initial population and front page.
- Population in row d is after night/day d.
- `action_next` is U_d, applied to the next front page and affecting row d+1.
- The final U is an action for an unobserved next day; do not treat it as causing the final recorded x.

Required paper/supplement comparisons:

1. MICRO versus each theory variant: target-share and coverage trajectories, with uncertainty for ensemble means.
2. Board-target share and activation/spending trajectories where the simulator exports them.
3. Signed theory-minus-MICRO residuals and mean-bias/RMSE, with the declared round window. The built-in error summary includes all exported rounds; recompute an explicitly defined late-window score if needed.
4. Across cells, matched (rho,b) regime maps with shared scales and residual panels. Average rounds 41–60 only if both sides reach that horizon; do not silently truncate one side.
5. Selected distribution/variance comparisons from raw paths. The built-in comparison command reports mean overlays; use your analysis engine for distribution distances and cluster-bootstrap intervals.

The package averages theory replicas within initial blocks and weights blocks equally. Use identical block weighting for MICRO. Do not give an initial condition more weight solely because more theory replicas were generated from it. The comparison command checks parameters, initial records, and complete block/round support before drawing overlays. Do not disable those checks merely to get a figure; fix or explicitly redesign the comparison.

## 5. Connect the paired-response tests already planned

Export actual pre-action snapshots from the existing replay utility. Run `branch` for theory and forced U=0/U=1 next-day rollouts for MICRO from the same complete snapshot. Both branches must include the next night and day. The reduced theory receives only that snapshot's projection and peer counts, which is exactly the closure being tested.

Use independent branch pairs and shared randomness where valid within each model; do not require microscopic and Langevin paths to share a noise realization. Compare raw continuous theory x_next against microscopic n_next/N for mean response, variance and distributional plots. At b=0 the theory branch contrast is exactly zero with common streams; verify the same causal null in MICRO without accidental RNG shifts.

The runner exports branch samples and paired chi_target, not T_pi/eta_IR estimates. To compute information, define a shared partition of [0,1] for both outputs and use the same conditioning, bins, snapshot weights, and bias/uncertainty procedure. Label the resulting T_pi as partition-dependent. If reconstructing a Pinsker numerator for a binned channel, use a bounded output statistic determined by those bins on both sides; keep the unbinned physical mean-response contrast separate. Never round the theory samples into binomial counts and call that the exact Langevin transition law.

A complete-snapshot comparison differs from a target-count-only empirical CMI. If mixing hidden snapshots into a coarse-conditioned kernel, mix the Q_u distributions using declared common weights before computing their JSD; an average of finely conditioned JSDs is a different quantity. Reuse the existing information engine rather than introducing inconsistent eta definitions.

## 6. Diagnose and report the residuals

If the matched reduced theory disagrees, use existing retained class vectors and IDs to test vote–evidence correlation, binomial count structure, diversity/overlap, and board/population covariance. A well-resolved model discrepancy is a valid result. Do not fit beta_evidence/beta_social separately for every cell: these are known simulator parameters. Any proposed new closure must be named, calibrated separately if necessary, and evaluated out of sample.

Distinguish implementation/adapter errors, sampling-law mismatch, numerical projection/step errors, Monte Carlo noise, and residual structural closure errors. Current code provides only the reduced theory; separately implemented class-density HMF results should remain a separate labeled model.

## Acceptance checklist and deliverables

- Package tests pass in the repository environment; keep a recorded version/dependency manifest.
- Resolved parameters and saved initial states match both sides exactly.
- One inspected episode confirms the day/action indexing and board stages.
- Both b=0 branch null and factual MICRO replay pass.
- Theory output is generated by the supplied Langevin runner, not copied from MICRO or old plots.
- At least one real matched cell has overlays, diagnostics, and a quantitative residual table before grid expansion.
- Output includes adapter code/tests, configuration/seed provenance, per-variant theory CSVs, matched simulator CSVs, plot PDFs/data, branch contrasts, and a concise scientific interpretation.
- Declare remaining approximations and any unsupported metric. No claim of phase transitions, exact HMF validation, or thermodynamic efficiency follows just from matching means.

Proceed with implementation and the small pilot using the existing authorized synthetic-simulation workflow. Do not change the production v3 game rules to force a match. The goal is a reproducible theory-prediction layer beside the simulation, so each paper plot can show what MICRO does, what the chosen theory predicts, and the uncertainty and size of their difference.
