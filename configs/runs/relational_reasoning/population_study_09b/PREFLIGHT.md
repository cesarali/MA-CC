# Study 09b preflight

Preflight date: 2026-08-29.

- Status: permitted
- Population size: `[12]`
- q values: `[2, 3]`
- L values: `[3]`
- Support redundancy: `[3]`
- Sensor size: `[6]`
- b values: `[9, 12]`
- Target semantics: false only
- Receiver dispositions: naive only
- Evidence strategies: strategic only
- Message modes: recommendation_plus_fact only
- beta: `[4.0]`
- theta: `[0.75]`
- Schedule: soft
- Frozen tasks: 2
- Repetitions: 1
- Structural regimes: `(2,9)`, `(2,12)`, `(3,9)`, `(3,12)`
- Cells and episodes: 8 and 8
- Provider calls: 1,056 nominal; 1,120 expected; 2,112 conservative
- Input tokens: 456,288 nominal; 483,752 expected; 912,576 conservative
- Output tokens: 1,056 nominal; 4,587,520 expected; 8,650,752 conservative
- Price: 0 proxy accounting units from current University model metadata
- Execution plan: 8 shards, throttle 8, 64 request permits, about 384 RPM
- Matched revised q=1 theory applicable: false

Task audit:

- `task_0001`: truth `SOUTHEAST`; false target `NORTH`; strategic real fact `f5`; fingerprint `1d10d7864cdbf01fe46afd34a5e16a3118e3d4720782a92d5a1ff5be571acac9`
- `task_0002`: truth `NORTH`; false target `NORTHWEST`; strategic real fact `f1`; fingerprint `edf1d1271acd4b71ffca9a8a920dc4fc4a01e70a95d8dec1ac64c7b3c909dbe0`

Machine-readable evidence is under `results/inspection/study09b_preflight/`.