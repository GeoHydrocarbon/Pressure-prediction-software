"""Extension boundaries must keep prediction algorithms independent of workflows."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from overpressure.algorithms.registry import ALGORITHM_SPECS, AlgorithmSpec, capabilities
from overpressure.config import Project
from overpressure.model import ModelBundle
from overpressure.modules.catalog import MODULES, module_for_page
from overpressure.modules.training import train


class ModuleContractTests(unittest.TestCase):
    def test_ui_configuration_does_not_import_torch(self):
        result=subprocess.run([sys.executable,'-c',
            "from overpressure.config import ALGORITHMS; from overpressure.model import ModelBundle; "
            "import sys; assert 'torch' not in sys.modules"],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_added_backend_trains_and_loads_without_changes_to_prediction_entry(self):
        # Simulate a future algorithm implementing the same interface as MLP.
        module=types.ModuleType('pressure_test_backend')
        class Predictor:
            def __init__(self,folder,device=None):
                self.folder=folder;self.metadata=json.loads((folder/'manifest.json').read_text())
                self.features=('DEN','Depth');self.feature_units={'DEN':'g/cc','Depth':'m'}
            def predict(self,features,batch_size=8192):
                x=np.asarray(features);return np.where(np.isfinite(x).all(axis=1),x[:,0]*2,np.nan)
        module.Predictor=Predictor
        module.train=lambda project,**kwargs: ('trained', project.settings['algorithm']['name'])
        spec=AlgorithmSpec('extension','Extension',(),
                           'pressure_test_backend:train','pressure_test_backend:Predictor')
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules,pressure_test_backend=module), patch.dict(ALGORITHM_SPECS,extension=spec):
            folder=Path(temp)
            (folder/'manifest.json').write_text(json.dumps(dict(schema_version=1,status='completed',algorithm='extension')))
            cfg=dict(algorithm=dict(name='extension'),physics=dict(enabled=False),data={})
            self.assertEqual(train(Project(folder/'project.yaml',cfg)),('trained','extension'))
            bundle=ModelBundle(folder)
            self.assertEqual(bundle.features,('DEN','Depth'))
            np.testing.assert_allclose(bundle.predict([[2.4,100],[np.nan,200]],batch_size=1),[4.8,np.nan])

    def test_planned_algorithms_fail_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            for name in ('rf','xgboost'):
                cfg=dict(algorithm=dict(name=name),physics=dict(enabled=False),data={})
                with self.assertRaisesRegex(ValueError,'尚未接入'):
                    train(Project(Path(temp)/'project.yaml',cfg))
            self.assertEqual(list(Path(temp).iterdir()),[])
        self.assertTrue(capabilities()['rf']['planned'])
        self.assertFalse(capabilities()['mlp']['planned'])

    def test_unknown_model_or_incomplete_package_is_never_loaded(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            for algorithm,status,message in [('unknown','completed','未知算法'),('mlp','failed','完成状态')]:
                (folder/'manifest.json').write_text(json.dumps(dict(schema_version=1,status=status,algorithm=algorithm)))
                with self.assertRaisesRegex(ValueError,message):ModelBundle(folder)

    def test_business_pages_have_unambiguous_modules(self):
        self.assertEqual([spec.key for spec in MODULES],['training','traditional','wells','seismic','results'])
        pages=[page for spec in MODULES for page in spec.pages]
        self.assertEqual(len(pages),len(set(pages)))
        self.assertEqual(module_for_page(1).key,'training')
        self.assertEqual(module_for_page(5).key,'seismic')
        self.assertEqual(module_for_page(7).key,'traditional')
        self.assertIsNone(module_for_page(2))  # project-owned time-depth relationship
