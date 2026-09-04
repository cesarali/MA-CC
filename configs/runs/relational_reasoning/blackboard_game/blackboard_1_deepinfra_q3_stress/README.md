# Blackboard 1 DeepInfra truthful false-q3 stress

Paid ten-round stress variant of the task_001 truthful strategic-report,
false-target `q=3` arm:

- 21 scientific cells (`rho = 0.74, 0.85, 1.00`; `b = 3, 6, 9, 12, 15, 18, 21`)
- 10 episodes per cell; 210 episodes total
- up to 5 active SLURM allocations, each bundling 2 independent cells
- 20 episode/request slots per full allocation (10 per scientific cell)
- 100 study-wide concurrent request slots
- 1,000-RPM study-wide dispatch ceiling
- Dashboard-semantic retained artifacts
- 3 prompt examples retained per cell

Credentials are loaded from the ignored repository-root `.env`:

```dotenv
DEEPINFRA_API_KEY=your-key-here
```

The controller targets `ALLOCATION_1`; task_001 truth is `ALLOCATION_0`.
Its frozen task-specific design contains 21 distinct, symbolically validated
target-preserving truthful reports. The study uses the generic planned
cell-array launcher. Generate its ten DeepInfra-compatible paired
initializations before submission.
