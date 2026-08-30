# Study 09h: DeepInfra DeepSeek V4 Flash 0731

This directory is a provider/model variant of `population_study_09h`. It keeps
the full false-target scientific design unchanged: frozen task 0002, `N=12`,
`L=3`, 30 rounds, strategic true evidence, naive receivers, `q={1,2}`,
`rho={0.80,0.85}`, `b={3,4,6,8,9,12}`, and 15 deterministic repetitions in
each of 24 cells (360 episodes total).

The only scientific implementation change is the LLM condition: DeepInfra
`deepseek-ai/DeepSeek-V4-Flash-0731`. Transport retries, scheduler topology,
and shared provider pacing are execution policy. No Study 09d episode is
reused because changing provider/model changes the measured scientific system.

NERSC production uses `mas-cc study prepare` with an external `/pscratch`
result root, followed by the generic detached supervisor in `scripts/nersc/`.
The checked-in `/work`, `all`, and `normal` fields remain Potsdam-safe source
defaults and are not used to request NERSC resources. See `PREFLIGHT_NERSC.md`
for the launch-specific audit.
