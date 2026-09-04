# Blackboard 1 DeepInfra smoke

This is a deliberately small, paid-provider smoke variant of the first
Blackboard study. It leaves the production `blackboard_1` study unchanged.

- Provider: DeepInfra
- Model: `deepseek-ai/DeepSeek-V4-Flash`
- Arm: no control
- Scientific cells: 3 (`rho = 0.74, 0.85, 1.00`)
- Episodes: 1 per cell, 3 total
- Population: 24
- Rounds: 5
- Maximum concurrent study requests: 3

The smoke checks DeepInfra authentication, model discovery, structured ballot
responses, Blackboard dynamics, checkpointing, and cell-array execution. It is
not a 100-concurrency load test.

## Credentials

Put the key in the ignored repository-root `.env` file. Never place it in YAML
or commit it to the repository:

```dotenv
DEEPINFRA_API_KEY=your-key-here
```

The normal DeepInfra URL is built into the provider. Only set
`DEEPINFRA_BASE_URL` when intentionally routing through another compatible
endpoint. The generic cell worker loads the repository-root `.env`; an already
exported environment variable takes precedence.

## Provider-free preflight

```bash
/home/ojedamarin/.local/share/miniforge3/bin/conda run -n MA-CC \
  mas-cc experiment preflight \
  --config configs/runs/relational_reasoning/blackboard_game/blackboard_1_deepinfra_smoke/blackboard_1_deepinfra_no_control.yaml \
  --output-dir /work/ojedamarin/Projects/LanguageGames/MA-CC/results/inspection/musr_blackboard_01_deepinfra_smoke_preflight
```

## Create the one paired DeepInfra initialization

The Potsdam-provider initialization cannot be reused because provider/model
identity is part of the artifact compatibility check. This command makes 24
paid initialization calls but does not run the five-round dynamics:

```bash
/home/ojedamarin/.local/share/miniforge3/bin/conda run -n MA-CC \
  mas-cc study initialize \
  --config-dir configs/runs/relational_reasoning/blackboard_game/blackboard_1_deepinfra_smoke \
  --output-dir /work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/musr_blackboard_01_deepinfra_smoke_initializations
```

## Submit after reviewing preflight and initialization

```bash
/home/ojedamarin/.local/share/miniforge3/bin/conda run -n MA-CC \
  mas-cc study submit \
  --config-dir configs/runs/relational_reasoning/blackboard_game/blackboard_1_deepinfra_smoke
```

No study-specific SLURM launcher is required.
