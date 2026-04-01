# sokoban Baseline Results

Benchmark version: `v1`

Objective gap column corresponds to `move_count_gap`.
Secondary objective gap column corresponds to `push_count_gap`.

| split | baseline | success_rate | mean_feasibility_score | mean_quality_score | mean_invalid_action_rate | mean_steps | mean_total_reward | mean_objective_gap | mean_secondary_objective_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | random_valid | 0.000 | 0.699 | 0.000 | 0.000 | 66.167 | 10.614 | 55.583 | -2.083 |
| test | push_when_possible | 0.000 | 0.740 | 0.000 | 0.000 | 90.750 | 15.937 | 80.167 | -3.083 |
| test | greedy_goal_progress | 0.083 | 0.784 | 0.078 | 0.000 | 110.833 | 20.556 | 100.250 | -3.250 |
| test | deadlock_avoiding_greedy | 0.083 | 0.784 | 0.078 | 0.000 | 110.833 | 20.556 | 100.250 | -3.250 |
| test | oracle | 1.000 | 1.000 | 0.941 | 0.000 | 10.583 | 5.955 | 0.000 | 0.000 |
| ood_more_boxes | random_valid | 0.000 | 0.691 | 0.000 | 0.000 | 72.167 | 11.615 | 56.667 | -1.750 |
| ood_more_boxes | push_when_possible | 0.000 | 0.691 | 0.000 | 0.000 | 51.833 | 7.832 | 36.333 | -1.417 |
| ood_more_boxes | greedy_goal_progress | 0.000 | 0.765 | 0.000 | 0.000 | 192.000 | 36.055 | 176.500 | -3.083 |
| ood_more_boxes | deadlock_avoiding_greedy | 0.000 | 0.765 | 0.000 | 0.000 | 192.000 | 36.055 | 176.500 | -3.083 |
| ood_more_boxes | oracle | 1.000 | 1.000 | 0.961 | 0.000 | 15.500 | 6.816 | 0.000 | 0.000 |
| ood_longer_plans | random_valid | 0.000 | 0.675 | 0.000 | 0.000 | 62.000 | 9.268 | 45.083 | -1.750 |
| ood_longer_plans | push_when_possible | 0.000 | 0.716 | 0.000 | 0.000 | 97.750 | 16.878 | 80.833 | -3.333 |
| ood_longer_plans | greedy_goal_progress | 0.000 | 0.765 | 0.000 | 0.000 | 192.000 | 35.848 | 175.083 | -4.250 |
| ood_longer_plans | deadlock_avoiding_greedy | 0.000 | 0.765 | 0.000 | 0.000 | 192.000 | 35.848 | 175.083 | -4.250 |
| ood_longer_plans | oracle | 1.000 | 1.000 | 0.964 | 0.000 | 16.917 | 7.103 | 0.000 | 0.000 |
