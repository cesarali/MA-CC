# MA-CC on Rutgers Amarel

Amarel uses the same deterministic study manifests and cell/config workers as
Potsdam. The only new layer is the site adapter: ordinary `sbatch` arrays on
account `general`, partition `main`, and QoS `normal`.

Prepare and submit from an Amarel checkout with the `MA-CC` Conda environment:

```bash
conda env update -n MA-CC -f environment.yml
conda run -n MA-CC python -m pip install -e .

conda run -n MA-CC mas-cc study submit \
  --execution-site amarel \
  --config-dir configs/runs/<family>/<study> \
  --results-dir /scratch/df630/MA-CC-results/studies/<study> \
  --require-results-under /scratch/df630/MA-CC-results
```

The submitter chooses one of the two generic templates in `scripts/Amarel/SLURM`.
It rejects cell plans above Amarel's hard 72-hour limit. Longer work must use
the existing episode checkpoints and resubmit the same deterministic study
root, with a provider-safe array throttle.

The jobs refuse cache paths outside `/scratch/df630`, check free scratch space,
and place the Conda environment/package cache, Hugging Face, Transformers,
Comet, and general application caches there. Results and scheduler logs must also stay below
`/scratch/df630/MA-CC-results`, never in the source checkout or home directory.

Use the local `amarel` courier for all remote commands. Do not SSH directly.
