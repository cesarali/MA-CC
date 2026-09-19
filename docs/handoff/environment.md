# The analysis environment the numbers were measured in

Every byte-identity claim made for the finalizer changes (#9, #10 and the fork's
`perf/*` branches) was measured in one environment: the shared mamba environment on the
Cygnus login node (`/shared/home/cesar/.local/share/mamba/envs/MA-CC`, Python 3.11.16) as
of 2026-09-19. `environments/cygnus-analysis.lock.txt` is its `pip list --format=freeze`
(184 pins). The load-bearing ones for the estimators:

| package | version | why it matters |
|---|---|---|
| numpy | 2.4.6 | pairwise summation order in `np.mean`, `np.log2`, the PCG64 stream |
| pandas | 3.0.5 | groupby ordering, `to_dict("records")` value types, Parquet writer |
| pyarrow | 25.0.0 | Parquet encoding of the published tables |
| scipy | 1.17.1 | checkpoint classifier |
| scikit-learn | 1.9.1 | checkpoint classifier |
| matplotlib | 3.11.1 | plots only |

"Byte-identical" means: same tables, same environment. A different numpy or pandas can
legitimately change the last bits of a bootstrap interval through summation order, and
that shows up in `compare_dirs.py` as a DIFFERENT with tiny deltas. When that happens,
check the environment first, the code second.

`environments/Dockerfile.analysis` builds a container from the lock for running the
finalizer, `cross_study`, `relocate` and the regression gate off-cluster (a workstation, a
CI runner, an Argo step). It is analysis-only: no provider credentials, no runner. Building
and pushing it to Harbor is a platform task, not part of this repository.

CI (`.github/workflows/tests.yml`) installs a subset of the same pins
(`requirements-ci.txt`); the lock is the superset the finalizer actually runs under.
