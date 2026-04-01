# symbolic_transformation Baseline Results

Benchmark version: `v1`

Objective gap column corresponds to `rewrite_step_gap`.

| split | baseline | success_rate | mean_feasibility_score | mean_quality_score | mean_invalid_action_rate | mean_steps | mean_total_reward | mean_objective_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arithmetic_test | random_valid | 0.667 | 0.921 | 0.407 | 0.000 | 8.833 | 3.483 | 4.083 |
| arithmetic_test | greedy_target_distance | 0.750 | 0.950 | 1.000 | 0.000 | 4.750 | 3.551 | 0.000 |
| arithmetic_test | greedy_expression_size | 0.750 | 0.950 | 1.000 | 0.000 | 4.750 | 3.515 | 0.000 |
| arithmetic_test | rule_priority | 0.750 | 0.950 | 1.000 | 0.000 | 4.750 | 3.572 | 0.000 |
| arithmetic_test | oracle | 0.750 | 0.950 | 1.000 | 0.000 | 4.750 | 3.609 | 0.000 |
| boolean_test | random_valid | 0.833 | 0.979 | 0.696 | 0.000 | 6.583 | 3.490 | 2.000 |
| boolean_test | greedy_target_distance | 0.917 | 0.983 | 1.000 | 0.000 | 4.583 | 3.637 | 0.000 |
| boolean_test | greedy_expression_size | 0.917 | 0.983 | 1.000 | 0.000 | 4.583 | 3.603 | 0.000 |
| boolean_test | rule_priority | 0.833 | 0.983 | 1.000 | 0.000 | 4.500 | 3.421 | -0.083 |
| boolean_test | oracle | 0.917 | 0.983 | 1.000 | 0.000 | 4.583 | 3.634 | 0.000 |
| ood_arithmetic_deeper | random_valid | 0.667 | 0.917 | 0.258 | 0.000 | 13.750 | 4.430 | 7.500 |
| ood_arithmetic_deeper | greedy_target_distance | 0.833 | 0.967 | 1.000 | 0.000 | 6.250 | 4.048 | 0.000 |
| ood_arithmetic_deeper | greedy_expression_size | 0.833 | 0.967 | 1.000 | 0.000 | 6.250 | 4.048 | 0.000 |
| ood_arithmetic_deeper | rule_priority | 0.833 | 0.967 | 1.000 | 0.000 | 6.250 | 4.044 | 0.000 |
| ood_arithmetic_deeper | oracle | 0.833 | 0.967 | 1.000 | 0.000 | 6.250 | 4.070 | 0.000 |
| ood_boolean_deeper | random_valid | 0.750 | 0.938 | 0.544 | 0.000 | 12.000 | 4.254 | 4.583 |
| ood_boolean_deeper | greedy_target_distance | 0.917 | 0.983 | 1.000 | 0.000 | 7.417 | 4.431 | 0.000 |
| ood_boolean_deeper | greedy_expression_size | 0.917 | 0.983 | 1.000 | 0.000 | 7.417 | 4.311 | 0.000 |
| ood_boolean_deeper | rule_priority | 0.917 | 0.983 | 1.000 | 0.000 | 7.417 | 4.319 | 0.000 |
| ood_boolean_deeper | oracle | 0.917 | 0.983 | 1.000 | 0.000 | 7.417 | 4.460 | 0.000 |
