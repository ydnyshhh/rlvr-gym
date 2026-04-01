# graph_planning Baseline Results

Benchmark version: `v1`

Objective gap column corresponds to `path_cost_gap`.

| split | baseline | success_rate | mean_feasibility_score | mean_quality_score | mean_invalid_action_rate | mean_steps | mean_total_reward | mean_objective_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | random_valid | 0.500 | 1.000 | 0.167 | 0.000 | 10.167 | -9.786 | 36.417 |
| test | greedy_low_cost | 0.167 | 1.000 | 0.167 | 0.000 | 13.500 | -2.682 | 14.250 |
| test | greedy_goal_distance | 1.000 | 1.000 | 0.917 | 0.000 | 1.667 | 1.281 | 0.167 |
| test | oracle | 1.000 | 1.000 | 1.000 | 0.000 | 1.667 | 1.317 | 0.000 |
| ood_larger_graphs | random_valid | 0.667 | 1.000 | 0.083 | 0.000 | 12.250 | -9.257 | 51.250 |
| ood_larger_graphs | greedy_low_cost | 0.083 | 1.000 | 0.083 | 0.000 | 22.167 | -1.940 | 22.000 |
| ood_larger_graphs | greedy_goal_distance | 1.000 | 1.000 | 0.667 | 0.000 | 1.583 | 0.960 | 1.500 |
| ood_larger_graphs | oracle | 1.000 | 1.000 | 1.000 | 0.000 | 2.000 | 1.380 | 0.000 |
| ood_directed_shift | random_valid | 1.000 | 1.000 | 0.083 | 0.000 | 6.833 | -2.526 | 30.000 |
| ood_directed_shift | greedy_low_cost | 0.167 | 1.000 | 0.000 | 0.000 | 17.083 | -3.865 | 31.667 |
| ood_directed_shift | greedy_goal_distance | 1.000 | 1.000 | 0.917 | 0.000 | 1.917 | 1.335 | 0.167 |
| ood_directed_shift | oracle | 1.000 | 1.000 | 1.000 | 0.000 | 1.917 | 1.364 | 0.000 |
