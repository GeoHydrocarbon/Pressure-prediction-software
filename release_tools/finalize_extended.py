"""Verify the declared runtime extension, source bytecode and smoke evidence, then ZIP."""
import argparse
import hashlib
import importlib.metadata
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
    for name in ('app','source','checks','baseline','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();app=args.app.resolve();source=args.source.resolve();workspace=Path(__file__).resolve().parents[1];out=args.output.resolve()
    if not out.is_relative_to(workspace/'outputs'):raise ValueError('Output outside workspace')
    m=json.loads((app/'source-manifest.json').read_text(encoding='utf-8'));checks=json.loads(args.checks.read_text(encoding='utf-8'))
    if checks['status']!='passed' or checks['gui']['version']!=m['version']:raise ValueError('Missing matching frozen acceptance evidence')
    if digest(args.baseline)!=m['baseline_runtime_sha256']:raise ValueError('Baseline checksum mismatch')
    actual={p.relative_to(app).as_posix():p for p in app.rglob('*') if p.is_file()}
    if set(actual)!=set(m['files'])|{'source-manifest.json'}:raise ValueError('Unexpected runtime file set')
    for file,sha in m['app_modules'].items():
        if digest(source/file)!=sha:raise ValueError('Source changed: '+file)
    packages={prefix:Path(importlib.metadata.distribution(dist).locate_file(prefix)) for prefix,dist in [('xgboost','xgboost'),('sklearn','scikit-learn'),('narwhals','narwhals')]}
    for file,sha in m['dependency_modules'].items():
        prefix,rest=file.split('/',1)
        if digest(packages[prefix]/rest)!=sha:raise ValueError('Dependency changed: '+file)
    for name in ('PressurePrediction.exe','PressureWorker.exe'):
        _,entries,_=unpack_executable(app/name)
        entry=next(e for e in entries if e['name']=='entry' and e['kind']==b's')
        expected=compile((source/'release_tools/entry.py').read_bytes(),'release_tools/entry.py','exec',dont_inherit=True)
        if marshal.loads(payload(entry))!=expected:raise ValueError('Launcher source mismatch')
        pyz=payload(next(e for e in entries if e['kind']==b'z'));toc=dict(marshal.loads(pyz[struct.unpack('!i',pyz[8:12])[0]:]))
        for file in (*m['app_modules'],*m['dependency_modules']):
            module='.'.join(Path(file).with_suffix('').parts).removesuffix('.__init__');_,offset,size=toc[module]
            origin=source/file if file in m['app_modules'] else packages[file.split('/',1)[0]]/file.split('/',1)[1]
            if marshal.loads(zlib.decompress(pyz[offset:offset+size]))!=compile(origin.read_bytes(),file,'exec',dont_inherit=True):raise ValueError('Frozen bytecode mismatch: '+module)
    cfg=yaml.safe_load((app/'_internal/project.example.yaml').read_text(encoding='utf-8'))
    if cfg['project_data']['wells'] or cfg['project_data']['well_assets'] or cfg['project_data']['horizons'] or cfg['workspace'] or cfg['rock_physics']:raise ValueError('Project template is not blank')
    if any(cfg['data'].get(k) for k in ('well_files','well_imports','development_wells','validation_wells','blind_wells')):raise ValueError('Private well reference in template')
    mutable={'PressurePrediction.exe','PressureWorker.exe','README.txt','build-info.txt','build-dependencies.txt','_internal/project.example.yaml','source-manifest.json'}
    prefixes=('_internal/xgboost/','_internal/sklearn/','_internal/narwhals/','_internal/overpressure/vendor/rockphypy/','docs/')
    def dependency_metadata(name):return name.startswith(('_internal/xgboost-','_internal/scikit_learn-','_internal/narwhals-')) and '.dist-info/' in name
    immutable=0
    with zipfile.ZipFile(args.baseline) as baseline:
        originals={e.filename[len('PressurePrediction/'):]:e for e in baseline.infolist() if e.filename.startswith('PressurePrediction/') and not e.is_dir()}
        for name,entry in originals.items():
            if name in mutable or name.startswith(prefixes) or dependency_metadata(name):continue
            if name not in m['files']:raise ValueError('Baseline dependency missing: '+name)
            with baseline.open(entry) as stream:old_hash=hashlib.file_digest(stream,'sha256').hexdigest()
            if old_hash!=m['files'][name]:raise ValueError('Undeclared baseline change: '+name)
            immutable+=1
        for name in actual.keys()-originals.keys():
            if name not in mutable and not name.startswith(prefixes) and not dependency_metadata(name):raise ValueError('Undeclared extra file: '+name)
    for name,file in actual.items():
        if name.startswith('_internal/overpressure/vendor/rockphypy/'):
            if digest(file)!=digest(source/'overpressure/vendor/rockphypy'/Path(name).name):raise ValueError('RockPhyPy resource mismatch')
        for prefix,directory in packages.items():
            resource='_internal/'+prefix+'/'
            if name.startswith(resource) and digest(file)!=digest(directory/name[len(resource):]):raise ValueError('Dependency resource mismatch: '+name)
    out.mkdir(parents=True,exist_ok=True);archive=out/('PressurePrediction-'+m['version']+'-Windows-x64-CPU.zip')
    if archive.exists():raise FileExistsError(archive)
    print('Verified sources, frozen bytecode, blank project and '+str(immutable)+' unchanged baseline files',flush=True)
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,file in sorted(actual.items()):
            sha=hashlib.sha256()
            with file.open('rb') as src,z.open('PressurePrediction/'+name,'w',force_zip64=True) as dst:
                for chunk in iter(lambda:src.read(1024*1024),b''):sha.update(chunk);dst.write(chunk)
            if name in m['files'] and sha.hexdigest()!=m['files'][name]:raise ValueError('Runtime changed during packaging: '+name)
    with zipfile.ZipFile(archive) as z:
        if z.testzip():raise ValueError('Archive CRC verification failed')
    checksum=digest(archive);(out/'SHA256SUMS.txt').write_text(checksum+'  '+archive.name+'\n',encoding='ascii')
    result=dict(version=m['version'],archive=str(archive),size_bytes=archive.stat().st_size,sha256=checksum,
        runtime='Windows x64 CPU',build=m['build'],dependency_versions=m['dependency_versions'],source_modules_verified=len(m['app_modules']),
        immutable_baseline_files_verified=immutable,baseline_runtime_sha256=m['baseline_runtime_sha256'],contains_user_data=False,blank_project=True,frozen_checks=checks['checks'])
    (out/'acceptance_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
