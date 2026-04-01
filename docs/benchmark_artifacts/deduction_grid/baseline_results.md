# deduction_grid Baseline Results

Benchmark version: `v1`

Objective gap column corresponds to `deduction_step_gap`.
Secondary objective gap column is unused for this family.

| split | baseline | success_rate | mean_feasibility_score | mean_quality_score | mean_invalid_action_rate | mean_steps | mean_total_reward | mean_objective_gap | mean_secondary_objective_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | random_valid | 1.000 | 1.000 | 1.000 | 0.000 | 14.917 | 4.834 | 0.000 | 0.000 |
| test | propagate_first | 1.000 | 1.000 | 1.000 | 0.000 | 14.917 | 4.834 | 0.000 | 0.000 |
| test | assert_first | 1.000 | 1.000 | 1.000 | 0.000 | 14.917 | 4.834 | 0.000 | 0.000 |
| test | rule_out_first | 1.000 | 1.000 | 1.000 | 0.000 | 14.917 | 4.834 | 0.000 | 0.000 |
| test | oracle | 1.000 | 1.000 | 1.000 | 0.000 | 14.917 | 4.834 | 0.000 | 0.000 |
| ood_more_entities | random_valid | 1.000 | 1.000 | 1.000 | 0.000 | 18.250 | 5.468 | 0.000 | 0.000 |
| ood_more_entities | propagate_first | 1.000 | 1.000 | 1.000 | 0.000 | 18.250 | 5.468 | 0.000 | 0.000 |
| ood_more_entities | assert_first | 1.000 | 1.000 | 1.000 | 0.000 | 18.250 | 5.468 | 0.000 | 0.000 |
| ood_more_entities | rule_out_first | 1.000 | 1.000 | 1.000 | 0.000 | 18.250 | 5.468 | 0.000 | 0.000 |
| ood_more_entities | oracle | 1.000 | 1.000 | 1.000 | 0.000 | 18.250 | 5.468 | 0.000 | 0.000 |
| ood_more_categories | random_valid | 1.000 | 1.000 | 1.000 | 0.000 | 19.833 | 5.768 | 0.000 | 0.000 |
| ood_more_categories | propagate_first | 1.000 | 1.000 | 1.000 | 0.000 | 19.833 | 5.768 | 0.000 | 0.000 |
| ood_more_categories | assert_first | 1.000 | 1.000 | 1.000 | 0.000 | 19.833 | 5.768 | 0.000 | 0.000 |
| ood_more_categories | rule_out_first | 1.000 | 1.000 | 1.000 | 0.000 | 19.833 | 5.768 | 0.000 | 0.000 |
| ood_more_categories | oracle | 1.000 | 1.000 | 1.000 | 0.000 | 19.833 | 5.768 | 0.000 | 0.000 |
