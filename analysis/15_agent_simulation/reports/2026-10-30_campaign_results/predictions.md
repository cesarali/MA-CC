| prediction | verdict | detail |
|---|---|---|
| P1 | met | task003_symmetric_15: 1->2, 2->3, 3->4, 4->6; task003_nosolution_uniform_15: 1->2, 2->3, 3->4, 4->6; op_symmetric_15: 1->2, 2->3, 3->4, 4->6; op_nosolution_uniform_15: 1->2, 2->3, 3->4, 4->6 |
| P7 | met | task003_symmetric_15: 1->2, 2->3, 3->4, 4->6; task003_nosolution_uniform_15: 1->2, 2->3, 4->6; op_symmetric_15: 1->2, 2->3, 3->4, 4->6; op_nosolution_uniform_15: 1->2, 2->3, 3->4, 4->6 |
| P2 | failed | task003_symmetric_15: 0 increase(s) of tau* with q; task003_nosolution_uniform_15: 0 increase(s) of tau* with q; op_symmetric_15: 2 increase(s) of tau* with q; op_nosolution_uniform_15: 1 increase(s) of tau* with q |
| P3 | met | task003 3->4: sym 2.84 vs no-sol 3.25; task003 4->6: sym 1.79 vs no-sol 1.84; op 3->4: sym 1.29 vs no-sol 3.71; op 4->6: sym 1.13 vs no-sol 2.75 |
| P4 | met | task003 higher in 6/6, task003 all >= 0.7: True; symmetric_15 rho 0.37: task003 0.93 vs OP 0.69; symmetric_15 rho 0.61: task003 1.01 vs OP 0.82; symmetric_15 rho 0.78: task003 0.83 vs OP -0.13; nosolution_uniform_15 rho 0.37: task003 0.89 vs OP 0.52; nosolution_uniform_15 rho 0.61: task003 1.10 vs OP 0.82; nosolution_uniform_15 rho 0.78: task003 0.94 vs OP 0.76 |
| P5 (4 rho values, 64 cells) | failed | G lower under dedup in 31% of 64 cells; mean relative change +10.6% |
| P6 (4 rho values, 64 cells) | failed | op_nosolution_uniform_15: 9/16 cells beyond 0.02; op_symmetric_15: 0/16 cells beyond 0.02; task003_nosolution_uniform_15: 0/16 cells beyond 0.02; task003_symmetric_15: 0/16 cells beyond 0.02 |
| P5 (3 rho values, 48 cells) | failed | G lower under dedup in 42% of 48 cells; mean relative change +3.8% |
| P6 (3 rho values, 48 cells) | failed | op_nosolution_uniform_15: 9/12 cells beyond 0.02; op_symmetric_15: 0/12 cells beyond 0.02; task003_nosolution_uniform_15: 0/12 cells beyond 0.02; task003_symmetric_15: 0/12 cells beyond 0.02 |
