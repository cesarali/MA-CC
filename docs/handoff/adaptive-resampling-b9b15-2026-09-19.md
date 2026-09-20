# Adaptive resampling vs fixed draws

Rows compared: 132 (requested 1000 draws each).
Draws used: mean 435, median 400, min 300, max 1000; 99 % of intervals stopped early; 43 % of the requested work done.
Endpoint shift (in units of the full-draw width, 132 intervals): low median 0.018 / p95 0.061 / max 0.170; high median 0.018 / p95 0.064 / max 0.097.
Point estimates identical: True; p-values identical: True.

| stage | fixed s | adaptive s |
|---|---|---|
| information_resampling | 25.0 | 16.0 |
| derived_study_aggregates | 143.2 | 143.2 |
| blackboard_calibration | 35.7 | 35.7 |
| total | 339.7 | 330.1 |

| metric | rows | draws mean | draws min | max low shift | max high shift |
|---|---|---|---|---|---|
| round_conditioning_state_count | 6 | 467 | 300 | 0.081 | 0.077 |
| round_controller_action_entropy | 6 | 417 | 300 | 0.074 | 0.005 |
| round_controller_action_entropy_given_population | 6 | 433 | 300 | 0.070 | 0.029 |
| round_dual_action_event_fraction | 6 | 467 | 300 | 0.170 | 0.033 |
| round_dual_action_state_fraction | 6 | 450 | 300 | 0.048 | 0.058 |
| round_epistemic_target_actuation_cmi | 6 | 417 | 300 | 0.020 | 0.034 |
| round_kappa_target_actuation_cmi | 6 | 450 | 300 | 0.077 | 0.097 |
| round_memory_target_actuation_cmi | 6 | 433 | 300 | 0.040 | 0.064 |
| round_phi_target_actuation_cmi | 6 | 417 | 300 | 0.029 | 0.037 |
| round_population_actuation_cmi | 6 | 383 | 300 | 0.033 | 0.040 |
| round_sensing_mi | 6 | 417 | 300 | 0.045 | 0.040 |
| round_sensor_mae | 6 | 483 | 400 | 0.052 | 0.038 |
| round_sensor_mse | 6 | 500 | 400 | 0.043 | 0.049 |
| round_single_action_slice_fraction | 6 | 400 | 400 | 0.053 | 0.034 |
| round_singleton_fraction | 6 | 467 | 300 | 0.071 | 0.073 |
| round_target_actuation_cmi | 6 | 400 | 300 | 0.025 | 0.063 |
| round_target_information_fraction | 6 | 483 | 300 | 0.059 | 0.063 |
| round_target_signed_actuation | 6 | 417 | 300 | 0.045 | 0.049 |
| round_target_signed_response_share | 6 | 467 | 300 | 0.037 | 0.047 |
| round_target_susceptibility | 6 | 317 | 300 | 0.062 | 0.064 |
| round_truth_actuation_cmi | 6 | 433 | 300 | 0.047 | 0.094 |
| round_truth_signed_actuation | 6 | 450 | 300 | 0.036 | 0.038 |
