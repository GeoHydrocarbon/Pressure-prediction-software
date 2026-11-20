from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import yaml
from overpressure.config import Project,validate_settings
from overpressure.data import prepare_data,audit_wells
from overpressure.well_workflow import read_well_input
from overpressure.well_io import selected_paths

ROOT=Path(__file__).resolve().parents[1]


class BatchWellTests(unittest.TestCase):
    def test_custom_training_ratio_controls_actual_sample_counts(self):
        self.project.settings['data'].update(training_selection='checked_files',train_ratio=.6)
        prepared=prepare_data(self.project)
        self.assertEqual(len(prepared.splits['train']),18);self.assertEqual(len(prepared.splits['validation']),12)

    def test_checked_files_override_all_legacy_well_and_blind_names(self):
        self.project.settings['data'].update(training_selection='checked_files',development_wells=['OLD'],blind_wells=['B'])
        prepared=prepare_data(self.project)
        self.assertEqual(set(prepared.splits['train'].well),{'A','V','B'})
        self.assertEqual(len(prepared.splits['blind']),0)
        self.assertEqual(len(prepared.splits['train']),24);self.assertEqual(len(prepared.splits['validation']),6)
        self.project.settings['data']['well_files']=['A.csv','V.csv']
        prepared=prepare_data(self.project)
        self.assertEqual(set(prepared.audit.frame.well),{'A','V'})
        self.assertNotIn('B',self.project.settings['data']['development_wells'])

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.folder=Path(self.tmp.name)
        cfg=yaml.safe_load((ROOT/'project.yaml').read_text(encoding='utf-8'))
        cfg["data"]["training_selection"]="named_wells"
        self.project=Project(self.folder/'project.yaml',cfg)
        for name,start in [('A',100),('V',200),('B',9000)]:
            frame=pd.DataFrame(dict(Depth=np.arange(start,start+10),AC=100,DEN=2.4,Ip=8000,Pp=np.arange(10)+20))
            # Missing well column intentionally exercises filename-derived well names.
            frame.to_csv(self.folder/(name+'.csv'),index=False)
        cfg['data'].update(well_files=['A.csv','V.csv','B.csv'],development_wells=['A','V'],validation_wells=[],blind_wells=['B'],split_method='random_points',train_ratio=.8)

    def test_batch_random_split_filename_wells_unique_ids_and_train_only_scaler(self):
        result=prepare_data(self.project)
        self.assertEqual(len(result.splits['train']),16);self.assertEqual(len(result.splits['validation']),4)
        self.assertFalse(result.audit.frame.source_row.duplicated().any())
        self.assertEqual(set(result.splits['blind'].well),{'B'})
        np.testing.assert_allclose(result.scaler.mean_,result.splits['train'][['Depth','AC','DEN','Ip']].mean())
        self.assertEqual(len(result.audit.report['sources']),3)
        self.assertEqual(result.audit.frame.source_file_row.tolist(),list(range(2,12))*3)

    def test_duplicate_well_files_are_rejected_and_empty_batch_has_no_legacy_fallback(self):
        frame=pd.read_csv(self.folder/'A.csv');frame['井']='A';frame.to_csv(self.folder/'copy.csv',index=False)
        self.project.settings['data']['well_files'].append('copy.csv')
        with self.assertRaisesRegex(ValueError,'同时出现'):audit_wells(self.project)
        self.project.settings['data']['well_files']=[]
        with self.assertRaisesRegex(ValueError,'至少选择'):audit_wells(self.project)

    def test_measured_chinese_fields_and_optional_prediction_labels(self):
        columns=self.project.settings['data']['columns'];units=self.project.settings['data']['units']
        frame=read_well_input([self.folder/'A.csv',self.folder/'V.csv'],columns,units)
        self.assertEqual(set(frame.well),{'A','V'});self.assertEqual(len(frame),20)
        path=self.folder/'points.csv';pd.DataFrame({'井名':['A','V'],'深度':[102,204],'实测压力':[31,42],'压力系数':[1.2,1.4]}).to_csv(path,index=False)
        points=read_well_input([path],dict(well='井',Depth='Depth',Pp='Pp'),dict(Depth='m',Pp='MPa'),measured=True)
        self.assertEqual(points.Pp.tolist(),[31,42]);self.assertEqual(points.well.tolist(),['A','V'])

    def test_leave_one_well_mode_validation_and_train_dispatch(self):
        cfg=self.project.settings;cfg['data']['split_method']='leave_one_well_out';validate_settings(cfg)
        preview=prepare_data(self.project);self.assertEqual(set(preview.splits['validation'].well),{'A'})
        from overpressure.training import train
        with patch('overpressure.cross_validation.cross_validate',return_value='cv') as cv:
            self.assertEqual(train(self.project),'cv');cv.assert_called_once()

    def test_batch_leave_one_well_and_final_refit_complete_without_blind_leakage(self):
        import json,io
        from contextlib import redirect_stdout
        from overpressure.training import train
        from overpressure.refit import refit
        cfg=self.project.settings;cfg['data']['split_method']='leave_one_well_out'
        cfg['algorithm'].update(hidden_sizes=[4],dropout=0);cfg['training'].update(epochs=1,batch_size=8)
        with redirect_stdout(io.StringIO()):
            cv=Path(train(self.project));final=Path(refit(self.project,cv))
        summary=json.loads((cv/'summary.json').read_text(encoding='utf-8'))
        self.assertEqual({row['validation_well'] for row in summary['folds']},{'A','V'})
        self.assertEqual(summary['pooled_metrics']['count'],20)
        manifest=json.loads((final/'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['purpose'],'final_refit');self.assertEqual(set(manifest['preprocessing']['fit_wells']),{'A','V'})


if __name__=='__main__':unittest.main()
