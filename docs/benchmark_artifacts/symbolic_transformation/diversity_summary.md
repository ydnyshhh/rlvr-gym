# symbolic_transformation Diversity Summary

Benchmark version: `v1`

| split | count | source_nodes_mean | target_nodes_mean | oracle_steps_mean | task_type_histogram |
| --- | --- | --- | --- | --- | --- |
| arithmetic_train | 24 | 12.580 | 4.080 | 4.500 | {"arithmetic_simplify": 24} |
| arithmetic_validation | 12 | 12.670 | 3.170 | 4.750 | {"arithmetic_simplify": 12} |
| arithmetic_test | 12 | 13.000 | 3.670 | 4.750 | {"arithmetic_simplify": 12} |
| boolean_train | 24 | 15.290 | 7.000 | 5.170 | {"boolean_nnf": 24} |
| boolean_validation | 12 | 15.830 | 7.170 | 4.750 | {"boolean_nnf": 12} |
| boolean_test | 12 | 12.750 | 4.420 | 4.580 | {"boolean_nnf": 12} |
| ood_arithmetic_deeper | 12 | 17.000 | 3.170 | 6.250 | {"arithmetic_simplify": 12} |
| ood_boolean_deeper | 12 | 19.000 | 5.920 | 7.420 | {"boolean_nnf": 12} |
