# paired_parent_exp_2: 60 additional parents per setting

Prepared 22 September 2026. This is batch 02 of the paired-parent experiment, not a third scientific design and not a replay of batch 01. No simulations have been launched.

Run-agent instructions: [RUN_AGENT.md](RUN_AGENT.md).

Entry point: [study.yaml](study.yaml), listing four run configs: q3_rho070.yaml, q3_rho100.yaml, q12_rho070.yaml, q12_rho100.yaml. Keep the entire bundle together; one run config covers only one setting. analysis.yaml preserves the original analysis recipe except expected_parents, now 240.

Each setting has 60 new parents, two uncontrolled preparation rounds, one saved checkpoint per parent, nine continuation branches, and ten rounds per branch. Preserve q={3,12}, rho={0.70,1.00}, budgets b={3,12}, N=24, and one continuation copy per branch. The nine branches are silence plus always/sensing crossed with truth/false at both budgets.

New root seed: 20260922 (original: 20260916). New study identity: blackboard-checkpoint-ensemble-02. New Potsdam output root: /work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/blackboard_checkpoint_ensemble_02.

Expected new data: 240 parents, 2,160 continuation paths, 21,600 continuation rounds, and 518,400 continuation micro-updates. Including the once-per-parent preparation gives 22,080 population rounds and 529,920 focal updates. With 24 initial agent calls per parent, the nominal no-retry demand is 535,680 provider requests. These are design counts, not observed results or a monetary quote.

The new batch plus the first gives 100 parent populations per setting if all new parents complete; older missing branches mean some combined comparisons have 98 rather than 100 complete pairs unless recovered separately.

Validation: all four run configs resolve with the repository's typed loader; core scientific sections were checked unchanged; new parent seed sets do not overlap the original sets under standalone and cell-0 derivations. Full credential-free preflight was attempted for all four and stopped because the original task file is absent locally. The runner must stage the exact original dataset and complete fresh preflight. See VALIDATION.json and CONFIG_CHANGES.json. Scheduler and budget limits were preserved; their adequacy for the larger workload is not certified by local validation.

The original source directory is /Users/rsanchez/Projects/MA-CC/configs/runs/relational_reasoning/blackboard_game/blackboard_checkpoint_ensemble_01. Its files were not changed. CONFIG_CHANGES.json records source hashes and every YAML field change. This package does not include the task dataset, credentials, a copied simulation pipeline, or a study-specific job script.
