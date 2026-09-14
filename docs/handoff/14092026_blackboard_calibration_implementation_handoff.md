# Blackboard calibration implementation handoff

The physical-cell `blackboard_calibration_v1` pipeline is integrated with standard
study aggregation. [Usage, source-field audit and scientific limits](../documentation/metrics/blackboard_calibration.md)
are documented together. No scientific study configuration was changed or live
study submitted; all execution evidence is from temporary mock fixtures.

The implementation adds action/exposure channel counts and parameters, empirical
and matched sampling-model exposure, direct and mixture active parameters,
block uncertainty, and opt-in within-cell held-out relaxation/susceptibility
reference predictions. Stable zero-compliance handling and signed affinity
boundaries are separate from legacy controlled-slot semantics.

Actual eligible controller/peer counts, actual board sample size, focal selection
and round update counts now survive retention. The audit established that
`controller_message_directly_exposed` marks transient direct recommendations,
not arbitrary sampled controller posts. Legacy board-message IDs still support
empirical calibration without the newly retained eligible composition.

The mock integration test executes a `results_only` board study, inspects the
published validation and analysis manifests, verifies count/estimate products
and archive contents, deletes the source run tree, and verifies offline
reaggregation reproduces the estimates. Requested and successful provider calls
in this test are local mock calls, not external LLM requests.

Validation commands use the existing local project Python:

```bash
/home/cesarali/miniconda3/envs/MA-CC/bin/python -m pytest \
  tests/mas_cc/analysis/test_blackboard_calibration.py \
  tests/mas_cc/test_causal_response.py \
  tests/mas_cc/test_single_affinity_consistency.py \
  tests/mas_cc/test_relational_blackboard.py \
  tests/mas_cc/test_results_only_resume.py \
  tests/mas_cc/test_relational_round_feedback_analysis.py \
  tests/mas_cc/test_blackboard_dashboard.py
```

Result: 173 passed. The new family contributes 20 tests, including mock
publication/reaggregation. A broader run including `test_studies.py` yielded
114 passes and seven failures due to pre-existing missing Study 06–08 and old
controlled-smoke config paths. Those failures occur before estimator execution.

Supported prediction claims are deliberately limited to held-out blocks within
physical cells. Cross-budget pooled training, equal-episode calibration and
samplers other than uniform without replacement are rejected rather than
silently implemented with different semantics. Changing boards carry
reference-only status, and exposed silence suppresses the unexposed-baseline
reference. No empirical scientific fit or cross-budget transfer result is claimed.

No study-specific SLURM job or replacement CMI implementation was added. Existing
user changes to theory files and documentation were preserved.
