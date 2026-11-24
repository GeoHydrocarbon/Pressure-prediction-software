"""Package only a checked frozen application and verify it against its source snapshot."""
import argparse
import hashlib
import importlib.util
import json
import marshal
from pathlib import Path
import struct
import zipfile
import zlib
import yaml
from rebuild_offline import digest,unpack_executable,payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--checks',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    app=args.app.resolve();source=args.source.resolve();output=args.output.resolve()
    workspace=Path(__file__).resolve().parents[1]
    if not output.is_relative_to(workspace/'outputs'):raise ValueError('Output outside workspace')
    output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((app/'source-manifest.json').read_text(encoding='utf-8'))
    checks=json.loads(args.checks.read_text(encoding='utf-8'))
    version=manifest['version']
    if checks['status']!='passed' or checks['gui']['version']!=version:raise ValueError('Missing or mismatched frozen checks')
    if digest(args.baseline)!=manifest['baseline_runtime_sha256']:raise ValueError('Baseline checksum mismatch')
    # Match each application module and the launcher to the current source, not
    # just the reported version. Dependencies must stay in the vetted runtime.
    for file,sha in manifest['app_modules'].items():
        if digest(source/file)!=sha:raise ValueError('Source snapshot mismatch: '+file)
    for name,sha in manifest['executable_sha256'].items():
        if digest(app/name)!=sha:raise ValueError('Executable checksum mismatch')
        _,entries,_=unpack_executable(app/name)
        entry=next(e for e in entries if e['name']=='entry' and e['kind']==b's')
        expected=compile((source/'release_tools'/'entry.py').read_bytes(),'release_tools/entry.py','exec',dont_inherit=True)
        if marshal.loads(payload(entry))!=expected:raise ValueError('Launcher source mismatch')
        pyz=payload(next(e for e in entries if e['kind']==b'z'))
        if pyz[4:8]!=importlib.util.MAGIC_NUMBER:raise ValueError('Bytecode mismatch')
        modules=dict(marshal.loads(pyz[struct.unpack('!i',pyz[8:12])[0]:]))
        for file in manifest['app_modules']:
            path=Path(file);module='.'.join(path.with_suffix('').parts).removesuffix('.__init__')
            _,offset,size=modules[module]
            code=marshal.loads(zlib.decompress(pyz[offset:offset+size]))
            expected=compile((source/file).read_bytes(),file,'exec',dont_inherit=True)
            if code!=expected:raise ValueError('Frozen module/source mismatch: '+module)
    template=app/'_internal'/'project.example.yaml'
    if digest(template)!=manifest['template_sha256']:raise ValueError('Template checksum mismatch')
    cfg=yaml.safe_load(template.read_text(encoding='utf-8'))
    assert not cfg['project_data']['wells'] and not cfg['project_data']['well_assets'] and not cfg['project_data']['horizons']
    assert not cfg['workspace'] and not cfg['traditional_prediction']
    assert not cfg['seismic']['density_file'] and not cfg['seismic']['impedance_file']
    assert not cfg['data']['well_files'] and not cfg['well_prediction']['measurement_files']
    assert cfg['output_dir']=='outputs'
    # Reject any extra file injected into the runtime; unchanged dependency
    # resources must have exactly the original content from the known ZIP.
    changed={'PressurePrediction.exe','PressureWorker.exe','README.txt','build-info.txt','_internal/project.example.yaml'}
    extras={'source-manifest.json'}
    with zipfile.ZipFile(args.baseline) as baseline:
        prefix='PressurePrediction/'
        originals={entry.filename[len(prefix):]:entry for entry in baseline.infolist()
                   if entry.filename.startswith(prefix) and not entry.is_dir()}
        actual={file.relative_to(app).as_posix():file for file in app.rglob('*') if file.is_file()}
        if set(actual)!=set(originals)|extras:raise ValueError('Unexpected or missing runtime files')
        for name,file in actual.items():
            if name in changed|extras:continue
            if file.stat().st_size!=originals[name].file_size:raise ValueError('Dependency size changed: '+name)
            with baseline.open(originals[name]) as old, file.open('rb') as new:
                if hashlib.file_digest(old,'sha256').digest()!=hashlib.file_digest(new,'sha256').digest():
                    raise ValueError('Dependency content changed: '+name)
    archive=output/f'PressurePrediction-{version}-Windows-x64-CPU.zip'
    if archive.exists():raise FileExistsError(archive)
    print('Verified empty project, source modules, launcher and runtime files',flush=True)
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,file in sorted(actual.items()):z.write(file,'PressurePrediction/'+name)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('ZIP integrity check failed')
    checksum=digest(archive)
    (output/'SHA256SUMS.txt').write_text(f'{checksum}  {archive.name}\n',encoding='ascii')
    result=dict(version=version,archive=str(archive),size_bytes=archive.stat().st_size,sha256=checksum,
                runtime='Windows x64 CPU',build=manifest['build'],source_modules_verified=len(manifest['app_modules']),
                baseline_runtime_sha256=manifest['baseline_runtime_sha256'],contains_user_data=False,blank_project=True,
                frozen_checks=checks['checks'])
    (output/'acceptance_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
