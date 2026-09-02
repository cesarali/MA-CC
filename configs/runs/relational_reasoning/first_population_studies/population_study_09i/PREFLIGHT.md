# Study 09i provider-free preflight

Preflight date: 2026-08-30. No completion, pricing, account, or scheduler call
was made. A provider-free zero-price planning quote was used only to exercise
the static estimator. Live availability/pricing/account checks and the
dedicated Potsdam `MA-CC` import check remain mandatory before submission.

## Resolved scientific contract

- Status: **PERMITTED**
- Strict contract: `relational_persistence_high_statistics_truth_v1`
- Frozen task: `n12_L3_r03_k3/task_0002`; truth and target both `NORTH`
- Fixed: `N=12`, `L=3`, rounds `30`, `q_c=6`, receiver `naive`,
  evidence `strategic`, `recommendation_plus_fact`, soft schedule, `beta=4.0`,
  `theta=0.75`
- Exact axes, in stable order:
  `q={1,2}` x `rho={0.80,0.85}` x `b={3,4,6,8,9,12}`
- Structural cells: `24`; target repetitions per cell: `15`; episodes: `360`
- Theory comparison: disabled; permanent analysis tables: CSV
- Controlled production updates contain one controller slot: q=1 leaves no
  ordinary peer and q=2 leaves exactly one ordinary peer

## Reuse audit

Only the eight q=2/strategic cells of Study 09e at `rho={0.80,0.85}` and
`b={3,6,9,12}` are scientifically eligible (at most 80 unique episodes).
Study 09g is excluded because L=2; false-target, neutral, q=1, new-budget, and
different-rho episodes are excluded.

The audit key is exact equality of task, q, L, rho, budget, evidence strategy,
target semantics, receiver disposition, message mode, beta, theta, rounds, and
controller semantics; seed-level duplicates count only once.

The Potsdam root is not readable from this machine because SSH authentication
is rejected, so **zero episodes are currently certified reusable**. The safe
new count is therefore 360. It falls to 280 only after all 80 candidates pass
sealed artifact, schema, seed, prompt, task, and resolved-condition validation.
Submission remains blocked pending that audit.

## Provider, token, cost, and runtime estimate

| Quantity | Unreused | With all 80 candidates validated |
|---|---:|---:|
| New episodes | 360 | 280 |
| Nominal provider calls | 133,920 | 104,160 |
| Expected provider calls | 141,120 | 109,760 |
| Conservative provider calls | 267,840 | 208,320 |
| Expected input tokens | 59,502,600 | 46,037,240 |
| Conservative input tokens | 112,950,720 | 87,389,760 |
| Expected output-token bound | 578,027,520 | 449,576,960 |
| Conservative output-token bound | 1,097,072,640 | 853,278,720 |
| Static rough runtime at 200 RPM | 11.76 h | 9.15 h |

Token estimates are deterministic planning bounds, not provider tokenizer
measurements. Planning cost is 0.0 `proxy_accounting_unit`, not a currency
price; a fresh live quote and account-limit check are required at launch.

## Execution plan and safe split

- 24 cell shards using the generic launcher; array throttle 2
- 10 CPUs, 12 GB, 16 hours, 10 episode slots, and 10 request permits per shard
- Effective ceiling 20 requests / 20 episode slots; declared target 500 RPM;
  planned sustained rate 266.7 RPM
- Expected call time about 8.82 hours at 266.7 RPM (6.86 hours if all 80
  candidates are later certified); static 200-RPM estimate 11.76 hours (9.15
  hours with full eligible reuse), before queueing or outages

Run this as an independent target-condition submission, sequentially with
Study 09h. The throttle remains unchanged; no higher-throttle assumption is
included in these estimates.
