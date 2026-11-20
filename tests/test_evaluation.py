import unittest
import numpy as np
import pandas as pd
from overpressure.evaluation import metrics, evaluation_table, paired_points


class EvaluationTests(unittest.TestCase):
    def test_known_metrics_and_zero_reference_mape_rule(self):
        result = metrics([0, 10, 20, np.nan], [1, 12, 16, 100])
        self.assertEqual(result["count"], 3)
        self.assertAlmostEqual(result["rmse"], np.sqrt(7))
        self.assertAlmostEqual(result["mae"], 7/3)
        self.assertAlmostEqual(result["mape"], 20)
        self.assertEqual(result["mape_count"], 2)
        self.assertEqual(result["mape_excluded_zero_count"], 1)
        self.assertAlmostEqual(result["r2"], 1-21/200)

    def test_constant_empty_and_negative_r2_are_not_fabricated(self):
        self.assertIsNone(metrics([0, 0], [1, 2])["mape"])
        self.assertIsNone(metrics([1, 1], [2, 3])["r2"])
        self.assertIsNone(metrics([], [])["rmse"])
        self.assertLess(metrics([1, 2], [10, 20])["r2"], 0)

    def test_table_and_scatter_pairs_exclude_invalid_labels_and_flags(self):
        frame = pd.DataFrame({"well": ["A", "A", "B", "B", "B"], "Pp": [0, 10, -1, 30, 50],
                              "Pp_pred": [1, 12, 2, np.nan, 52], "supervised_valid": [True, True, True, True, False]})
        pairs = paired_points(frame); table = evaluation_table(frame)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(table.iloc[0]["count"], 2)
        self.assertEqual(table.iloc[0]["mape_count"], 1)
        self.assertEqual(table.iloc[2]["count"], 0)
        self.assertAlmostEqual(table.iloc[0]["mape"], 20)

    def test_overall_is_pooled_not_average_of_well_scores(self):
        frame = pd.DataFrame({"well": ["A", "A", "A", "B"], "Pp": [10]*4, "Pp_pred": [10, 10, 10, 20]})
        table = evaluation_table(frame)
        self.assertEqual(table.iloc[0]["mae"], 2.5)
        self.assertEqual(table.iloc[0]["rmse"], 5)
