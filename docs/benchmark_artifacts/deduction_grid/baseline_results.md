# deduction_grid Baseline Results

Benchmark version: `v1`

Objective gap column corresponds to `deduction_step_gap`.

| split | baseline | success_rate | mean_feasibility_score | mean_quality_score | mean_invalid_action_rate | mean_steps | mean_total_reward | mean_objective_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | random_valid | 1.000 | 1.000 | 0.679 | 0.000 | 9.500 | 3.799 | -4.500 |
| test | propagate_first | 1.000 | 1.000 | 0.321 | 0.000 | 4.500 | 2.843 | -9.500 |
| test | assert_first | 1.000 | 1.000 | 1.000 | 0.000 | 14.000 | 4.660 | 0.000 |
| test | rule_out_first | 1.000 | 1.000 | 0.458 | 0.000 | 6.417 | 3.209 | -7.583 |
| test | oracle | 1.000 | 1.000 | 1.000 | 0.000 | 14.000 | 4.660 | 0.000 |
| ood_more_entities | random_valid | 1.000 | 1.000 | 0.745 | 0.000 | 12.667 | 4.402 | -4.333 |
| ood_more_entities | propagate_first | 1.000 | 1.000 | 0.279 | 0.000 | 4.750 | 2.889 | -12.250 |
| ood_more_entities | assert_first | 1.000 | 1.000 | 1.000 | 0.000 | 17.000 | 5.230 | 0.000 |
| ood_more_entities | rule_out_first | 1.000 | 1.000 | 0.412 | 0.000 | 7.000 | 3.319 | -10.000 |
| ood_more_entities | oracle | 1.000 | 1.000 | 1.000 | 0.000 | 17.000 | 5.230 | 0.000 |
| ood_more_categories | random_valid | 1.000 | 1.000 | 0.741 | 0.000 | 13.333 | 4.529 | -4.667 |
| ood_more_categories | propagate_first | 1.000 | 1.000 | 0.292 | 0.000 | 5.250 | 2.985 | -12.750 |
| ood_more_categories | assert_first | 1.000 | 1.000 | 1.000 | 0.000 | 18.000 | 5.420 | 0.000 |
| ood_more_categories | rule_out_first | 1.000 | 1.000 | 0.449 | 0.000 | 8.083 | 3.526 | -9.917 |
| ood_more_categories | oracle | 1.000 | 1.000 | 1.000 | 0.000 | 18.000 | 5.420 | 0.000 |
