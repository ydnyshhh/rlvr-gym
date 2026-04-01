# scheduling Baseline Results

Benchmark version: `v1`

Objective gap column corresponds to `tardiness_gap`.
Secondary objective gap column is unused for this family.

| split | baseline | success_rate | mean_feasibility_score | mean_quality_score | mean_invalid_action_rate | mean_steps | mean_total_reward | mean_objective_gap | mean_secondary_objective_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | random_ready | 1.000 | 1.000 | 0.167 | 0.000 | 6.500 | 2.720 | 3.417 | 0.000 |
| test | shortest_processing_time | 1.000 | 1.000 | 0.500 | 0.000 | 6.500 | 2.860 | 1.333 | 0.000 |
| test | earliest_deadline | 1.000 | 1.000 | 0.500 | 0.000 | 6.500 | 2.876 | 1.083 | 0.000 |
| test | minimum_slack | 1.000 | 1.000 | 0.417 | 0.000 | 6.500 | 2.858 | 1.417 | 0.000 |
| test | oracle | 1.000 | 1.000 | 1.000 | 0.000 | 6.500 | 2.956 | 0.000 | 0.000 |
| ood_more_jobs | random_ready | 1.000 | 1.000 | 0.083 | 0.000 | 9.000 | 3.089 | 6.167 | 0.000 |
| ood_more_jobs | shortest_processing_time | 1.000 | 1.000 | 0.167 | 0.000 | 9.000 | 3.193 | 3.417 | 0.000 |
| ood_more_jobs | earliest_deadline | 1.000 | 1.000 | 0.417 | 0.000 | 9.000 | 3.273 | 1.500 | 0.000 |
| ood_more_jobs | minimum_slack | 1.000 | 1.000 | 0.250 | 0.000 | 9.000 | 3.260 | 1.750 | 0.000 |
| ood_more_jobs | oracle | 1.000 | 1.000 | 1.000 | 0.000 | 9.000 | 3.344 | 0.000 | 0.000 |
| ood_tighter_constraints | random_ready | 1.000 | 1.000 | 0.083 | 0.000 | 8.000 | 2.955 | 3.333 | 0.000 |
| ood_tighter_constraints | shortest_processing_time | 1.000 | 1.000 | 0.583 | 0.000 | 8.000 | 3.016 | 2.000 | 0.000 |
| ood_tighter_constraints | earliest_deadline | 1.000 | 1.000 | 0.583 | 0.000 | 8.000 | 3.065 | 0.667 | 0.000 |
| ood_tighter_constraints | minimum_slack | 1.000 | 1.000 | 0.583 | 0.000 | 8.000 | 3.062 | 0.750 | 0.000 |
| ood_tighter_constraints | oracle | 1.000 | 1.000 | 1.000 | 0.000 | 8.000 | 3.098 | 0.000 | 0.000 |
