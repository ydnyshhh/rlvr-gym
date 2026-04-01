# deduction_grid Baseline Results

Benchmark version: `v1`

Objective gap column corresponds to `deduction_step_gap`.

| split | baseline | success_rate | mean_feasibility_score | mean_quality_score | mean_invalid_action_rate | mean_steps | mean_total_reward | mean_objective_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | random_valid | 0.917 | 0.979 | 0.452 | 0.000 | 27.083 | 6.882 | 13.083 |
| test | propagate_first | 1.000 | 1.000 | 1.000 | 0.000 | 2.000 | 2.380 | -12.000 |
| test | assert_first | 1.000 | 1.000 | 1.000 | 0.000 | 14.000 | 4.660 | 0.000 |
| test | rule_out_first | 1.000 | 1.000 | 0.000 | 0.000 | 38.000 | 9.202 | 24.000 |
| test | oracle | 1.000 | 1.000 | 1.000 | 0.000 | 14.000 | 4.660 | 0.000 |
| ood_more_entities | random_valid | 0.750 | 0.938 | 0.294 | 0.000 | 42.417 | 9.284 | 25.417 |
| ood_more_entities | propagate_first | 1.000 | 1.000 | 1.000 | 0.000 | 2.000 | 2.380 | -15.000 |
| ood_more_entities | assert_first | 1.000 | 1.000 | 1.000 | 0.000 | 17.000 | 5.230 | 0.000 |
| ood_more_entities | rule_out_first | 0.000 | 0.750 | 0.000 | 0.000 | 60.000 | 10.333 | 43.000 |
| ood_more_entities | oracle | 1.000 | 1.000 | 1.000 | 0.000 | 17.000 | 5.230 | 0.000 |
| ood_more_categories | random_valid | 1.000 | 1.000 | 0.296 | 0.000 | 38.500 | 9.302 | 20.500 |
| ood_more_categories | propagate_first | 1.000 | 1.000 | 1.000 | 0.000 | 2.000 | 2.380 | -16.000 |
| ood_more_categories | assert_first | 1.000 | 1.000 | 1.000 | 0.000 | 18.000 | 5.420 | 0.000 |
| ood_more_categories | rule_out_first | 1.000 | 1.000 | 0.000 | 0.000 | 50.000 | 11.482 | 32.000 |
| ood_more_categories | oracle | 1.000 | 1.000 | 1.000 | 0.000 | 18.000 | 5.420 | 0.000 |
