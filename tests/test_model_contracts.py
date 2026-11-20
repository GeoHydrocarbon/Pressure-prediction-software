from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import yaml

from overpressure.config import FEATURES
from overpressure.data import sha256_file
from overpressure.model import ModelBundle, PointMLP, eaton_pressure, loss_terms


ROOT = Path(__file__).resolve().parents[1]


class ModelContractTests(unittest.TestCase):
    def setUp(self):
        self.physics = yaml.safe_load((ROOT / "project.yaml").read_text(encoding="utf-8"))["physics"]

    def test_point_input_and_no_window(self):
        model = PointMLP([8], 0)
        self.assertEqual(tuple(model(torch.ones(3, 4)).shape), (3,))
        with self.assertRaisesRegex(ValueError, "窗口"):
            model(torch.ones(3, 10, 4))

    def test_disabled_physics_does_not_compute_eaton(self):
        self.physics.update(enabled=False, nac_b=1e9)
        loss, data, physical = loss_terms(torch.tensor([20.]), torch.tensor([30.]), torch.tensor([[2000., 0., 2.4, 8000.]]), self.physics)
        self.assertEqual(loss.item(), 100)
        self.assertEqual(physical.item(), 0)

    def test_physics_weight_and_gradients(self):
        raw = torch.tensor([[2000., 100., 2.4, 8000.]])
        self.physics.update(enabled=True, weight=.25)
        pred = torch.tensor([25.], requires_grad=True)
        loss, data, physical = loss_terms(pred, torch.tensor([30.]), raw, self.physics)
        torch.testing.assert_close(loss, .75 * data + .25 * physical)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        altered = deepcopy(self.physics); altered["nac_b"] += .2
        self.assertNotEqual(eaton_pressure(raw, self.physics).item(), eaton_pressure(raw, altered).item())

    def test_bundle_round_trip_invalid_points_and_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            model = PointMLP([8], 0).eval()
            torch.save(model.state_dict(), folder / "weights.pt")
            manifest = {"schema_version": 1, "status": "completed", "algorithm": "mlp", "features": list(FEATURES),
                        "architecture": {"hidden_sizes": [8], "dropout": 0},
                        "weights_sha256": sha256_file(folder / "weights.pt"),
                        "preprocessing": {"mean": [0] * 4, "scale": [1] * 4}}
            (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            x = np.array([[100, 100, 2.4, 8000], [100, np.nan, 2.4, 8000], [200, 100, 2.4, 8000]])
            pred = ModelBundle(folder, device="cpu").predict(x)
            with torch.inference_mode():
                expected = model(torch.tensor(x[[0, 2]], dtype=torch.float32)).numpy()
            np.testing.assert_allclose(pred[[0, 2]], expected)
            self.assertTrue(np.isnan(pred[1]))
            with (folder / "weights.pt").open("ab") as f:
                f.write(b"altered")
            with self.assertRaisesRegex(ValueError, "校验"):
                ModelBundle(folder, device="cpu")


if __name__ == "__main__":
    unittest.main()
