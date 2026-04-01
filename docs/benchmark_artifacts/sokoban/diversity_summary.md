# sokoban Diversity Summary

Benchmark version: `v1`

| split | count | num_boxes_mean | board_height_mean | board_width_mean | oracle_steps_mean | oracle_push_count_mean | boxes_on_goals_at_start_mean | unsolved_boxes_at_start_mean | box_interaction_pair_count_mean | box_interaction_component_count_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 24 | 2.000 | 8.210 | 8.210 | 10.250 | 4.250 | 0.000 | 2.000 | 0.960 | 1.040 |
| validation | 12 | 2.000 | 8.330 | 8.330 | 10.170 | 4.420 | 0.000 | 2.000 | 1.000 | 1.000 |
| test | 12 | 2.000 | 8.420 | 8.420 | 10.580 | 4.080 | 0.000 | 2.000 | 1.000 | 1.000 |
| ood_more_boxes | 12 | 3.000 | 8.670 | 8.670 | 15.500 | 4.670 | 0.420 | 2.580 | 2.000 | 1.250 |
| ood_longer_plans | 12 | 3.000 | 8.580 | 8.580 | 16.920 | 5.330 | 0.330 | 2.670 | 1.750 | 1.500 |
