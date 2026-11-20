import unittest
import numpy as np
import pandas as pd
from overpressure.well_points import match_pressure_points, well_roles


class WellPointTests(unittest.TestCase):
    def test_exact_interpolation_no_extrapolation_and_unknown_well(self):
        curve = pd.DataFrame({"well": ["NEW"]*3, "Depth": [100., 110., 120.], "Pp_pred": [10., 20., 30.]})
        points = pd.DataFrame({"well": ["NEW"]*4 + ["OTHER"], "Depth": [100., 105., 120., 125., 105.], "Pp": [10., 14., 31., 32., 12.]})
        result = match_pressure_points(curve, points)
        self.assertEqual(result.match_status.tolist(), ["matched_exact", "matched_interpolated", "matched_exact", "outside_range", "well_not_found"])
        self.assertEqual(result.Pp_pred.iloc[1], 15)
        self.assertEqual(int(result.supervised_valid.sum()), 3)

    def test_invalid_points_and_large_gaps_are_never_bridged(self):
        curve = pd.DataFrame({"well": ["A"]*5, "Depth": [100., 110., 120., 130., 500.], "Pp_pred": [1., np.nan, 3., 4., 5.]})
        points = pd.DataFrame({"well": ["A"]*4, "Depth": [105., 110., 125., 200.], "Pp": [1., 2., 3., 4.]})
        result = match_pressure_points(curve, points)
        self.assertEqual(result.match_status.tolist(), ["gap", "invalid_curve_point", "matched_interpolated", "gap"])
        self.assertEqual(match_pressure_points(curve, points, max_gap_m=4).match_status.iloc[2], "gap")

    def test_duplicates_bad_pressure_and_empty_reference(self):
        curve = pd.DataFrame({"well": ["A"]*4, "Depth": [100., 110., 110., 120.], "Pp_pred": [1., 2., 3., 4.]})
        points = pd.DataFrame({"well": ["A"]*3, "Depth": [110., 105., 115.], "Pp": [2., -1., np.nan]})
        result = match_pressure_points(curve, points)
        self.assertEqual(result.match_status.tolist(), ["ambiguous_depth", "invalid_measurement", "invalid_measurement"])
        self.assertTrue(match_pressure_points(curve, points.iloc[:0]).empty)

    def test_known_training_or_selection_wells_cannot_be_declared_independent(self):
        metadata = {"preprocessing": {"fit_wells": ["TRAIN"]}, "project": {"data": {"development_wells": ["TRAIN", "VAL"]}}}
        roles = well_roles(metadata, ["TRAIN", "VAL", "NEW"], True)
        self.assertEqual(roles["TRAIN"], "training_review")
        self.assertEqual(roles["VAL"], "development_review")
        self.assertEqual(roles["NEW"], "independent_validation_user_confirmed")
        self.assertEqual(well_roles({}, ["NEW"], True)["NEW"], "independence_unverified")
