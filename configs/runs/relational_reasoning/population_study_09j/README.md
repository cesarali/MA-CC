# Study 09j: N=24 false-target q=1, L=2 persistence

Study 09j is the larger-population, high-statistics false-target successor to
Study 09f. It preserves the simple `q=1`, `L=2`, 30-round mechanism, but it is
**not an exact replication**. The receiver changes from `naive` to `vigilant`,
while evidence remains `strategic`; the intended condition is therefore
`vigilant_strategic`.

The frozen task is `pop24_L2_r06/task_0002`. Its truth is `SOUTHWEST`; the
controller targets the false option `SOUTH`. Every controller fact is a real
true task fact selected by the existing strategic evidence policy and checked
by the existing citation rules.

Each of 20 repetitions has one natural large-language-model-generated initial
population state. That complete physical state—votes, reasons, exposed facts,
active and historical facts, task, and agent/fact assignment—is persisted once
and reused across all 18 `rho x b` cells and Study 09k. Persistence `rho` and
budget `b` begin acting only after this shared state exists.
