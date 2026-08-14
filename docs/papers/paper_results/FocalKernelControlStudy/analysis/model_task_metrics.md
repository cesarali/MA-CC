# Per-model and per-task statistics

Rates pool the six matched social-context buckets within each task. Coverage is `n_valid / n_expected`; incomplete coverage should be considered when comparing models.

| Model | Task | Valid/expected | Coverage | Control adoption | Truth | Stay | Switch |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-4o | 1: Evacuation | 60/60 | 1.000 | 0.450 | 0.283 | 0.317 | 0.683 |
| GPT-4o | 4: Traffic accident | 60/60 | 1.000 | 0.467 | 0.800 | 0.367 | 0.633 |
| GPT-4o | 9: Hospital transfer | 60/60 | 1.000 | 0.417 | 0.917 | 0.383 | 0.617 |
| GPT-4o | 13: Laboratory theft | 60/60 | 1.000 | 0.533 | 0.533 | 0.300 | 0.700 |
| GPT-4o | 16: Backup datacenter | 60/60 | 1.000 | 0.417 | 0.683 | 0.467 | 0.533 |
| GPT-4o | 23: Banquet venue | 60/60 | 1.000 | 0.483 | 0.800 | 0.400 | 0.600 |
| GPT-4o | 27: Research station | 60/60 | 1.000 | 0.417 | 0.733 | 0.350 | 0.650 |
| GPT-4o | 30: Lead investor | 60/60 | 1.000 | 0.400 | 0.867 | 0.567 | 0.433 |
| GPT-4o | 36: Datacenter migration | 59/60 | 0.983 | 0.492 | 0.814 | 0.153 | 0.847 |
| GPT-4o | 41: Space evacuation | 60/60 | 1.000 | 0.483 | 0.717 | 0.217 | 0.783 |
| GPT-5 Mini | 1: Evacuation | 60/60 | 1.000 | 0.583 | 0.417 | 0.267 | 0.733 |
| GPT-5 Mini | 4: Traffic accident | 60/60 | 1.000 | 0.467 | 0.917 | 0.417 | 0.583 |
| GPT-5 Mini | 9: Hospital transfer | 60/60 | 1.000 | 0.483 | 0.983 | 0.317 | 0.683 |
| GPT-5 Mini | 13: Laboratory theft | 60/60 | 1.000 | 0.567 | 0.667 | 0.233 | 0.767 |
| GPT-5 Mini | 16: Backup datacenter | 60/60 | 1.000 | 0.550 | 0.767 | 0.350 | 0.650 |
| GPT-5 Mini | 23: Banquet venue | 60/60 | 1.000 | 0.417 | 0.683 | 0.417 | 0.583 |
| GPT-5 Mini | 27: Research station | 60/60 | 1.000 | 0.467 | 0.767 | 0.267 | 0.733 |
| GPT-5 Mini | 30: Lead investor | 60/60 | 1.000 | 0.450 | 0.850 | 0.450 | 0.550 |
| GPT-5 Mini | 36: Datacenter migration | 60/60 | 1.000 | 0.583 | 0.517 | 0.067 | 0.933 |
| GPT-5 Mini | 41: Space evacuation | 60/60 | 1.000 | 0.333 | 0.417 | 0.350 | 0.650 |
| GPT-OSS 120B | 1: Evacuation | 60/60 | 1.000 | 0.517 | 0.467 | 0.333 | 0.667 |
| GPT-OSS 120B | 4: Traffic accident | 60/60 | 1.000 | 0.467 | 0.900 | 0.367 | 0.633 |
| GPT-OSS 120B | 9: Hospital transfer | 60/60 | 1.000 | 0.500 | 1.000 | 0.300 | 0.700 |
| GPT-OSS 120B | 13: Laboratory theft | 60/60 | 1.000 | 0.617 | 0.650 | 0.233 | 0.767 |
| GPT-OSS 120B | 16: Backup datacenter | 60/60 | 1.000 | 0.550 | 0.767 | 0.350 | 0.650 |
| GPT-OSS 120B | 23: Banquet venue | 60/60 | 1.000 | 0.433 | 0.833 | 0.450 | 0.550 |
| GPT-OSS 120B | 27: Research station | 60/60 | 1.000 | 0.533 | 0.817 | 0.183 | 0.817 |
| GPT-OSS 120B | 30: Lead investor | 60/60 | 1.000 | 0.467 | 0.933 | 0.517 | 0.483 |
| GPT-OSS 120B | 36: Datacenter migration | 60/60 | 1.000 | 0.600 | 0.633 | 0.117 | 0.883 |
| GPT-OSS 120B | 41: Space evacuation | 60/60 | 1.000 | 0.517 | 0.500 | 0.083 | 0.917 |
| Gemma4 31B | 1: Evacuation | 60/60 | 1.000 | 0.683 | 0.517 | 0.200 | 0.800 |
| Gemma4 31B | 4: Traffic accident | 45/60 | 0.750 | 0.578 | 1.000 | 0.378 | 0.622 |
| Gemma4 31B | 9: Hospital transfer | 60/60 | 1.000 | 0.483 | 0.983 | 0.317 | 0.683 |
| Gemma4 31B | 13: Laboratory theft | 59/60 | 0.983 | 0.695 | 0.525 | 0.271 | 0.729 |
| Gemma4 31B | 16: Backup datacenter | 60/60 | 1.000 | 0.500 | 0.633 | 0.450 | 0.550 |
| Gemma4 31B | 23: Banquet venue | 60/60 | 1.000 | 0.517 | 0.600 | 0.350 | 0.650 |
| Gemma4 31B | 27: Research station | 60/60 | 1.000 | 0.467 | 0.733 | 0.317 | 0.683 |
| Gemma4 31B | 30: Lead investor | 60/60 | 1.000 | 0.450 | 0.883 | 0.533 | 0.467 |
| Gemma4 31B | 36: Datacenter migration | 60/60 | 1.000 | 0.567 | 0.683 | 0.183 | 0.817 |
| Gemma4 31B | 41: Space evacuation | 60/60 | 1.000 | 0.467 | 0.833 | 0.133 | 0.867 |
| Kimi K2.6 | 1: Evacuation | 12/60 | 0.200 | 0.500 | 0.917 | 0.500 | 0.500 |
| Kimi K2.6 | 4: Traffic accident | 23/60 | 0.383 | 0.478 | 1.000 | 0.348 | 0.652 |
| Kimi K2.6 | 9: Hospital transfer | 52/60 | 0.867 | 0.481 | 1.000 | 0.346 | 0.654 |
| Kimi K2.6 | 13: Laboratory theft | 10/60 | 0.167 | 0.900 | 0.600 | 0.100 | 0.900 |
| Kimi K2.6 | 16: Backup datacenter | 33/60 | 0.550 | 0.576 | 0.939 | 0.394 | 0.606 |
| Kimi K2.6 | 23: Banquet venue | 18/60 | 0.300 | 0.500 | 0.944 | 0.444 | 0.556 |
| Kimi K2.6 | 27: Research station | 25/60 | 0.417 | 0.520 | 0.960 | 0.000 | 1.000 |
| Kimi K2.6 | 30: Lead investor | 44/60 | 0.733 | 0.477 | 1.000 | 0.523 | 0.477 |
| Kimi K2.6 | 36: Datacenter migration | 11/60 | 0.183 | 0.727 | 1.000 | 0.000 | 1.000 |
| Kimi K2.6 | 41: Space evacuation | 17/60 | 0.283 | 0.471 | 0.588 | 0.000 | 1.000 |
| Qwen3 30B A3B | 1: Evacuation | 60/60 | 1.000 | 0.467 | 0.617 | 0.333 | 0.667 |
| Qwen3 30B A3B | 4: Traffic accident | 60/60 | 1.000 | 0.483 | 0.817 | 0.483 | 0.517 |
| Qwen3 30B A3B | 9: Hospital transfer | 58/60 | 0.967 | 0.241 | 0.724 | 0.483 | 0.517 |
| Qwen3 30B A3B | 13: Laboratory theft | 60/60 | 1.000 | 0.350 | 0.383 | 0.633 | 0.367 |
| Qwen3 30B A3B | 16: Backup datacenter | 60/60 | 1.000 | 0.350 | 0.517 | 0.583 | 0.417 |
| Qwen3 30B A3B | 23: Banquet venue | 60/60 | 1.000 | 0.333 | 0.600 | 0.500 | 0.500 |
| Qwen3 30B A3B | 27: Research station | 60/60 | 1.000 | 0.600 | 0.617 | 0.400 | 0.600 |
| Qwen3 30B A3B | 30: Lead investor | 60/60 | 1.000 | 0.267 | 0.767 | 0.717 | 0.283 |
| Qwen3 30B A3B | 36: Datacenter migration | 60/60 | 1.000 | 0.300 | 0.550 | 0.517 | 0.483 |
| Qwen3 30B A3B | 41: Space evacuation | 60/60 | 1.000 | 0.283 | 0.150 | 0.300 | 0.700 |
