from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from overpressure import runtime


class RuntimeTests(unittest.TestCase):
    def test_source_worker_arguments_preserve_paths(self):
        with patch.object(runtime.sys, 'frozen', False, create=True), patch.object(runtime, 'worker_python', return_value='python.exe'):
            self.assertEqual(runtime.worker_command('overpressure',['train','--project','中文 项目.yaml']),
                             ('python.exe',['-u','-m','overpressure','train','--project','中文 项目.yaml']))

    def test_frozen_worker_does_not_use_conda_or_python_m(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/'PressureWorker.exe').touch()
            with patch.object(runtime.sys,'frozen',True,create=True), patch.object(runtime,'application_root',return_value=root):
                exe,args=runtime.worker_command('overpressure.slice_worker',['--index','中文 路径.json'])
                self.assertEqual(Path(exe),root/'PressureWorker.exe')
                self.assertEqual(args,['--pressure-worker','overpressure.slice_worker','--index','中文 路径.json'])

    def test_default_frozen_project_is_created_once_outside_bundle(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); bundle=root/'bundle'; bundle.mkdir(); user=root/'user'; user.mkdir()
            (bundle/'project.example.yaml').write_text('example',encoding='utf-8')
            with patch.object(runtime.sys,'frozen',True,create=True), patch.object(runtime.sys,'_MEIPASS',str(bundle),create=True), patch.object(runtime,'user_data_root',return_value=user):
                project=runtime.default_project()
                self.assertEqual(project,user/'project.yaml')
                self.assertEqual(project.read_text(encoding='utf-8'),'example')
                project.write_text('user edits',encoding='utf-8')
                runtime.default_project()
                self.assertEqual(project.read_text(encoding='utf-8'),'user edits')
