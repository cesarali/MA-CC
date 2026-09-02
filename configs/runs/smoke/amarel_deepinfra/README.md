# Amarel DeepInfra smoke

This is the smallest real end-to-end Amarel study. It contains one scientific
cell and one episode of the two-agent toy coordination game. The episode has
two interactions and a conservative limit of ten DeepInfra requests.

Submit only through the Amarel site adapter, with results outside the checkout:

```bash
mas-cc study submit \
  --execution-site amarel \
  --config-dir configs/runs/smoke/amarel_deepinfra \
  --results-dir /scratch/df630/MA-CC-results/studies/amarel-deepinfra-smoke \
  --require-results-under /scratch/df630/MA-CC-results
```

The generated plan must contain one cell shard, throttle one, one CPU, 2 GB of
memory, and a ten-minute walltime. The generic Amarel launcher performs the
scratch-space/cache checks and invokes the ordinary MA-CC cell worker.
