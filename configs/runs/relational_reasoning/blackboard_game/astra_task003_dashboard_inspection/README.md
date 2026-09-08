# ASTRA task-003 dashboard inspection

Small real-model version of `astra_task003_false_control_30x30` for checking the
study, cell, and episode dashboard.

## Design

- University provider with `gwdg/openai-gpt-oss-120b`.
- MuSR task 003, candidate 130.
- False-target adaptive-communication controller.
- 24 agents and 4 rounds. The calibrated truthful-selective task loader requires
  the retained `private/N24_assignment.json` artifact and rejects other
  population sizes.
- 3 repetitions per cell.
- Persistence `rho`: `0.70`, `0.85`.
- Controller post budget `b`: `2`, `4`.
- 4 cells and 12 episodes in total.
- Two concurrent local episode workers and at most two simultaneous provider requests.
- `dashboard_semantic` storage, which retains the public information required by the dashboard without retaining full private prompts and responses.

Each episode has 24 initialization decisions, 96 population-update decisions,
and up to 4 controller-policy decisions. The corrected preflight reports 1,488
lower-bound calls, 1,752 expected calls, and 4,464 conservative calls including
the retry bound. The configured hard ceiling is 5,000 requests.

## Local workflow

From the repository root, with `POTSDAM_API_KEY` and
`BASE_POTSDAM_LLM_URL` available in the environment:

```text
mas-cc experiment preflight \
  --config configs/runs/relational_reasoning/blackboard_game/astra_task003_dashboard_inspection/false_control_llm.yaml \
  --output-dir results/inspection/astra_task003_dashboard_inspection_preflight_n24

mas-cc experiment run \
  --config configs/runs/relational_reasoning/blackboard_game/astra_task003_dashboard_inspection/false_control_llm.yaml \
  --output-dir results/local/astra_task003_dashboard_inspection_n24 \
  --approve-preflight "$(cat results/inspection/astra_task003_dashboard_inspection_preflight_n24/preflight_id.txt)"

mas-cc blackboard dashboard \
  --study-dir results/local/astra_task003_dashboard_inspection_n24
```

Preflight performs no model calls. The experiment command does make real model
calls and should only be run after reviewing its preflight report and cost
limits.

The `study.yaml` file is included for the standard Potsdam workflow. It uses the
generic launcher and does not add a study-specific SLURM job file.
