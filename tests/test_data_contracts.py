from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

from overpressure.config import Project, load_project, unit_factor, validate_settings
from overpressure.data import canonical_features, feature_validity, prepare_data
from overpressure.time_depth import TimeDepth


ROOT = Path(__file__).resolve().parents[1]


class DataContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.cfg = yaml.safe_load((ROOT / "project.yaml").read_text(encoding="utf-8"))
        self.cfg["data"].pop("well_files",None)
        self.cfg["data"].pop("well_imports",None)
        self.cfg["data"]["training_selection"]="named_wells"
        self.cfg["data"].update(well_file="wells.csv", split_method="well_holdout", development_wells=["A", "V"], validation_wells=["V"], blind_wells=["B"])
        self.cfg["data"]["columns"]["well"] = "well"
        self.cfg["data"]["depth_reference"] = {"confirmed": True, "type": "TVD", "datum": "海平面"}
        self.cfg["time_depth"].update(confirmed=True, depth_type="TVD", coefficients=[100, 1000], valid_time_range=[0, 2], datum="海平面", source="测试")
        self.table = pd.DataFrame({"well": ["A", "A", "V", "B"], "Depth": [100, 200, 500, 1000],
                                   "AC": [100, 90, 80, 70], "DEN": [2.2, 2.3, 2.4, 2.5],
                                   "Ip": [6000, 7000, 8000, 9000], "Pp": [10, 20, 30, 40]})

    def project(self):
        self.table.to_csv(self.folder / "wells.csv", index=False)
        validate_settings(self.cfg)
        return Project(self.folder / "project.yaml", deepcopy(self.cfg))

    def random_project(self):
        self.cfg["data"].update(split_method="random_points", validation_wells=[])
        self.table = pd.DataFrame({"well": ["A"] * 50 + ["V"] * 50 + ["B"] * 3,
                                   "Depth": np.arange(103) + 100, "AC": 100., "DEN": 2.4, "Ip": 8000., "Pp": 10.})
        return self.project()

    def test_random_split_is_reproducible_disjoint_and_excludes_blind(self):
        project = self.random_project(); result = prepare_data(project); again = prepare_data(project)
        self.assertEqual([len(result.splits[r]) for r in ("train", "validation", "blind")], [80, 20, 3])
        train_rows, val_rows = (set(result.splits[r].source_row) for r in ("train", "validation"))
        self.assertFalse(train_rows & val_rows)
        self.assertEqual(train_rows | val_rows, set(range(2, 102)))
        for role in ("train", "validation"):
            self.assertEqual(set(result.splits[role].well), {"A", "V"})
            self.assertEqual(result.splits[role].source_row.tolist(), again.splits[role].source_row.tolist())
        self.cfg["training"]["seed"] += 1
        changed = prepare_data(self.project())
        self.assertNotEqual(train_rows, set(changed.splits["train"].source_row))

    def test_random_scaler_ignores_validation_and_blind_features(self):
        baseline = prepare_data(self.random_project())
        rows = baseline.audit.frame.loc[baseline.audit.frame.role.isin(["validation", "blind"])].index
        self.table.loc[rows, ["Depth", "AC", "DEN", "Ip"]] *= 100
        self.table.loc[self.table.well == "B", "Pp"] = np.nan
        after = prepare_data(self.project())
        np.testing.assert_array_equal(baseline.scaler.mean_, after.scaler.mean_)
        np.testing.assert_array_equal(baseline.scaler.scale_, after.scaler.scale_)
        self.assertEqual(baseline.splits["train"].source_row.tolist(), after.splits["train"].source_row.tolist())

    def test_random_filters_before_split_and_rejects_empty_partition(self):
        self.random_project(); self.table.loc[0, "Pp"] = np.nan; self.table.loc[1, "DEN"] = -1
        prepared = prepare_data(self.project())
        self.assertEqual(len(prepared.splits["train"]), 78)
        self.assertEqual(len(prepared.splits["validation"]), 20)
        self.assertEqual(prepared.audit.frame.role.iloc[:2].tolist(), ["excluded_invalid"] * 2)
        self.cfg["data"]["train_ratio"] = .001
        with self.assertRaisesRegex(ValueError, "监督样本"):
            prepare_data(self.project())

    def test_invalid_random_configuration_is_rejected(self):
        self.random_project()
        for ratio in (0, 1, np.nan, True):
            self.cfg["data"]["train_ratio"] = ratio
            with self.assertRaisesRegex(ValueError, "比例"):
                validate_settings(self.cfg)

    def test_blind_and_validation_do_not_change_scaler(self):
        before = prepare_data(self.project())
        self.table.loc[self.table.well != "A", ["Depth", "AC", "DEN", "Ip"]] *= 1000
        after = prepare_data(self.project())
        np.testing.assert_array_equal(before.scaler.mean_, after.scaler.mean_)
        np.testing.assert_array_equal(before.scaler.scale_, after.scaler.scale_)
        self.assertEqual(set(after.splits["train"].well), {"A"})
        self.assertEqual(set(after.splits["blind"].well), {"B"})

    def test_missing_label_does_not_remove_predictable_point(self):
        self.table.loc[0, "Pp"] = np.nan
        result = prepare_data(self.project())
        row = result.audit.frame.iloc[0]
        self.assertTrue(row.prediction_valid)
        self.assertFalse(row.supervised_valid)
        self.assertEqual(result.audit.issues.iloc[0].source_row, 2)
        self.assertEqual(len(result.splits["train"]), 1)

    def test_missing_blind_labels_do_not_block_training(self):
        baseline = prepare_data(self.project())
        self.table.loc[self.table.well == "B", "Pp"] = np.nan
        result = prepare_data(self.project())
        self.assertTrue(result.splits["blind"].empty)
        np.testing.assert_array_equal(result.scaler.mean_, baseline.scaler.mean_)

    def test_first_point_and_point_after_invalid_input_remain_usable(self):
        features = canonical_features([100, 200, 300], [2.4, np.nan, 2.5], [8000, 9000, 10000])
        self.assertEqual(feature_validity(features).tolist(), [True, False, True])

    def test_units_and_shared_ac_formula(self):
        features = canonical_features([2000], np.array([2400]) * unit_factor("DEN", "kg/m3"),
                                      np.array([8000000]) * unit_factor("Ip", "kg/m3*m/s"))
        np.testing.assert_allclose(features[0], [2000, 91.44, 2.4, 8000])
        self.assertAlmostEqual(unit_factor("AC", "us/m"), .3048)
        self.assertAlmostEqual(unit_factor("Pp", "kPa"), .001)

    def test_overlap_and_unsupported_physics_are_rejected(self):
        self.cfg["data"]["blind_wells"] = ["A"]
        with self.assertRaisesRegex(ValueError, "盲井"):
            validate_settings(self.cfg)
        self.cfg["data"]["blind_wells"] = ["B"]
        self.cfg["algorithm"]["name"] = "rf"
        self.cfg["physics"]["enabled"] = True
        with self.assertRaisesRegex(ValueError, "不支持"):
            validate_settings(self.cfg)

    def test_unassigned_well_is_excluded(self):
        self.table.loc[len(self.table)] = ["UNKNOWN", 200, 100, 2.4, 8000, 10]
        prepared = prepare_data(self.project())
        self.assertEqual(prepared.audit.report["unassigned_rows"], 1)
        self.assertFalse(any("UNKNOWN" in set(s.well) for s in prepared.splits.values()))

    def test_all_invalid_rows_of_a_well_are_rejected(self):
        self.table.loc[self.table.well == "V", "Pp"] = np.nan
        with self.assertRaisesRegex(ValueError, "有效"):
            prepare_data(self.project())

    def test_polynomial_confirmation_units_and_range(self):
        td = TimeDepth(self.project())
        np.testing.assert_allclose(td.transform([0, 1, 2], input_convention="twt"), [100, 1100, 2100])
        np.testing.assert_allclose(td.transform([3],input_convention='twt'),[3100])
        with self.assertRaisesRegex(ValueError, "类型"):
            td.transform([1], input_convention="owt")
        self.cfg["time_depth"]["confirmed"] = False
        np.testing.assert_allclose(TimeDepth(self.project()).transform([1], input_convention="twt"), [1100])
        self.cfg["time_depth"].update(confirmed=True, time_unit="ms", depth_unit="ft", coefficients=[100, 1], valid_time_range=[0, 2000])
        np.testing.assert_allclose(TimeDepth(self.project()).transform([1], input_convention="twt"), [1100 * .3048])

    def test_polynomial_interior_turning_point_is_rejected(self):
        # Derivative = (t-1)^2 - .1: positive at endpoints, negative inside.
        self.cfg["time_depth"]["coefficients"] = [100, .9, -1, 1 / 3]
        with self.assertRaisesRegex(ValueError, "单调"):
            TimeDepth(self.project()).transform([0,2],input_convention='twt')

    def test_depth_metadata_and_confirmations_do_not_block_conversion(self):
        self.cfg["time_depth"].update(depth_type="MD", datum="", source="", confirmed=False)
        self.cfg["data"]["depth_reference"] = {"confirmed": False, "type": "unknown", "datum": "different"}
        np.testing.assert_allclose(TimeDepth(self.project()).transform([1], input_convention="twt"), [1100])
        self.cfg["data"].pop("depth_reference")
        self.cfg["time_depth"].pop("confirmed")
        self.cfg["time_depth"].pop("depth_type")
        np.testing.assert_allclose(TimeDepth(self.project()).transform([1], input_convention="twt"), [1100])

    def test_table_interpolation_mask_and_unsorted_rejection(self):
        table = pd.DataFrame({"time": [0, 1, 2], "depth": [100, 1100, 3100]})
        table.to_csv(self.folder / "td.csv", index=False)
        self.cfg["time_depth"].update(method="table", table_file="td.csv", extrapolation="mask")
        td = TimeDepth(self.project())
        result = td.transform([.5, 3], input_convention="twt")
        self.assertAlmostEqual(result[0], 600)
        self.assertTrue(np.isnan(result[1]))
        table.iloc[::-1].to_csv(self.folder / "td.csv", index=False)
        with self.assertRaisesRegex(ValueError, "递增"):
            TimeDepth(self.project())

    def test_project_relative_path_is_independent_of_working_directory(self):
        path = self.folder / "project.yaml"
        path.write_text(yaml.safe_dump(self.cfg, allow_unicode=True), encoding="utf-8")
        self.assertEqual(load_project(path).resolve("wells.csv"), self.folder / "wells.csv")

    def test_old_project_depth_fields_are_ignored_and_cli_audit_still_works(self):
        from contextlib import redirect_stdout
        import io
        from overpressure.__main__ import audit
        self.project()
        path = self.folder / "project.yaml"
        path.write_text(yaml.safe_dump(self.cfg, allow_unicode=True), encoding="utf-8")
        loaded = load_project(path)
        self.assertNotIn("depth_reference", loaded.settings["data"])
        self.assertNotIn("confirmed", loaded.settings["time_depth"])
        with redirect_stdout(io.StringIO()): _, report = audit(path, wells_only=True)
        self.assertEqual(report["status"], "completed")
        self.assertNotIn("time_depth_confirmed", report)


if __name__ == "__main__":
    unittest.main()
