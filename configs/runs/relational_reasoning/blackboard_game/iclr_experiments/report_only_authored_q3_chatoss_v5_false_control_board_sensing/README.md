# report_only_authored_q3_chatoss_v5_false_control_board_sensing

Prompt-v5 Task-003 false-control study using delayed public-board controller
sensing.

## Scientific design

- arm: false control only
- social sampling budget: `q=3`
- controller board-sensing budget: `q_c=12` public messages per night
- Day 1: uncontrolled
- Night k: sample the completed Day-k board and prepare `U_(k+1)`
- participant public actions: REPORT or NONE
- controller public actions: LLM-authored grounded REPORT only
- citation scope: active memory plus grounded REPORT facts sampled now
- epistemic persistence: 0.70, 0.85, 1.00
- intervention budgets: 6, 12, 18
- rounds: 30
- repetitions per cell: 60

## Size

- cells: 9
- episodes: 540

The study reuses the source v5 family's paired initialization artifacts:

`/shared/home/cesar/work/results/studies/report_only_authored_q3_q12_chatoss_v5_initializations`

No paid provider run is authorized by these files.
