"""Extend a SHA256-verified frozen CPU runtime with explicit audited resources.

Unlike rebuild_offline.py, this builder adds the new XGBoost dependency and
RockPhyPy source/licence files. No dependency is silently borrowed at execution
time. A complete per-file manifest and frozen smoke checks are mandatory.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import marshal
from pathlib import Path
import shutil
import struct
import sys
import zipfile
import zlib

from rebuild_offline import digest,unpack_executable,payload,ENTRY,COOKIE,MAGIC


def rewrite(path,source,packages):
    bootloader,entries,library=unpack_executable(path);app_files={};dependency_files={}
    replaced=[]
    for entry in entries:
        if entry['kind']==b'z':
            blob=payload(entry)
            if blob[:4]!=b'PYZ\0' or blob[4:8]!=importlib.util.MAGIC_NUMBER:raise ValueError('Python bytecode version mismatch')
            toc=dict(marshal.loads(blob[struct.unpack('!i',blob[8:12])[0]:]))
            modules={name:(kind,blob[offset:offset+size]) for name,(kind,offset,size) in toc.items()
                     if not any(name==p or name.startswith(p+'.') for p in ('overpressure',*packages))}
            for directory,prefix,manifest in [(source/'overpressure','overpressure',app_files)]+[(directory,prefix,dependency_files) for prefix,directory in packages.items()]:
                for file in sorted(directory.rglob('*.py')):
                    if '__pycache__' in file.parts:continue
                    relative=Path(prefix)/file.relative_to(directory);name='.'.join(relative.with_suffix('').parts)
                    is_package=file.name=='__init__.py'
                    if is_package:name=name.removesuffix('.__init__')
                    code=compile(file.read_bytes(),relative.as_posix(),'exec',dont_inherit=True)
                    modules[name]=(int(is_package),zlib.compress(marshal.dumps(code),6));manifest[relative.as_posix()]=digest(file)
            output=io.BytesIO();output.write(b'\0'*17);new_toc=[]
            for name,(kind,data) in sorted(modules.items()):new_toc.append((name,(kind,output.tell(),len(data))));output.write(data)
            offset=output.tell();output.write(marshal.dumps(new_toc));output.seek(0)
            output.write(b'PYZ\0'+importlib.util.MAGIC_NUMBER+struct.pack('!i',offset));data=output.getvalue();replaced.append('PYZ')
        elif entry['kind']==b's' and entry['name']=='entry':
            data=marshal.dumps(compile((source/'release_tools/entry.py').read_bytes(),'release_tools/entry.py','exec',dont_inherit=True));replaced.append('entry')
        else:continue
        entry.update(data=zlib.compress(data,9) if entry['compressed'] else data,original=len(data))
    if sorted(replaced)!=['PYZ','entry']:raise ValueError('Unexpected executable archive')
    data=io.BytesIO();toc=[]
    for entry in entries:
        offset=data.tell();data.write(entry['data']);name=entry['name'].encode('utf-8')+b'\0';length=(ENTRY.size+len(name)+15)//16*16
        toc.append(ENTRY.pack(length,offset,len(entry['data']),entry['original'],entry['compressed'],entry['kind'])+name.ljust(length-ENTRY.size,b'\0'))
    offset=data.tell();table=b''.join(toc);data.write(table);data.write(COOKIE.pack(MAGIC,data.tell()+COOKIE.size,offset,len(table),sys.version_info.major*100+sys.version_info.minor,library))
    path.write_bytes(bootloader+data.getvalue())
    _,checked,_=unpack_executable(path)
    for entry in checked:
        if len(payload(entry))!=entry['original']:raise ValueError('Invalid regenerated executable')
    return app_files,dependency_files


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--runtime',type=Path,required=True);parser.add_argument('--sha256',required=True)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    workspace=Path(__file__).resolve().parents[1];source=args.source.resolve();output=args.output.resolve()
    if not output.is_relative_to(workspace/'outputs') or output.exists():raise ValueError('Use a new output directory inside workspace outputs')
    if digest(args.runtime)!=args.sha256:raise ValueError('Baseline checksum mismatch')
    distributions={name:importlib.metadata.distribution(name) for name in ('xgboost','scikit-learn','narwhals')}
    if distributions['xgboost'].version!='3.3.0' or distributions['scikit-learn'].version!='1.9.0':raise ValueError('Unexpected tree dependency version')
    packages={name:Path(distributions[distribution].locate_file(name)).resolve() for name,distribution in [('xgboost','xgboost'),('sklearn','scikit-learn'),('narwhals','narwhals')]}
    output.mkdir(parents=True)
    with zipfile.ZipFile(args.runtime) as archive:
        for entry in archive.infolist():
            if not (output/entry.filename).resolve().is_relative_to(output):raise ValueError('Unsafe baseline member')
        archive.extractall(output)
    app=output/'PressurePrediction';old_manifest=app/'source-manifest.json'
    if old_manifest.exists():old_manifest.unlink()
    app_files,dependency_files=rewrite(app/'PressurePrediction.exe',source,packages)
    if (app_files,dependency_files)!=rewrite(app/'PressureWorker.exe',source,packages):raise ValueError('Executable module mismatch')
    def copy(file,target):target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,target)
    internal=(app/'_internal').resolve()
    for prefix in packages:
        stale=(internal/prefix).resolve()
        if stale.exists():
            if stale.parent!=internal or not stale.is_relative_to(output):raise ValueError('Unsafe dependency replacement')
            shutil.rmtree(stale)
    for old in internal.glob('scikit_learn-*.dist-info'):
        if old.resolve().parent!=internal or not old.resolve().is_relative_to(output):raise ValueError('Unsafe metadata replacement')
        shutil.rmtree(old)
    for prefix,package in packages.items():
        for file in package.rglob('*'):
            if file.is_file() and '__pycache__' not in file.parts and file.suffix!='.pyc':copy(file,internal/prefix/file.relative_to(package))
    for distribution in distributions.values():
        metadata=Path(distribution._path)
        for file in metadata.rglob('*'):
            if file.is_file() and file.name not in ('direct_url.json','RECORD','INSTALLER','REQUESTED'):copy(file,internal/metadata.name/file.relative_to(metadata))
    vendor=source/'overpressure/vendor/rockphypy'
    for file in vendor.iterdir():
        if file.is_file():copy(file,app/'_internal/overpressure/vendor/rockphypy'/file.name)
    for file in (source/'docs').glob('*.md'):copy(file,app/'docs'/file.name)
    copy(source/'project.example.yaml',app/'_internal/project.example.yaml');copy(source/'release_tools/README-windows.txt',app/'README.txt')
    version={};exec((source/'overpressure/__init__.py').read_text(encoding='utf-8'),version);version=version['__version__']
    versions={name:d.version for name,d in distributions.items()}
    (app/'build-info.txt').write_text(f'version={version}\nruntime=Windows x64 CPU\nbuild=audited CPU runtime extension\n'+''.join(k+'='+v+'\n' for k,v in versions.items()),encoding='utf-8')
    dependencies=app/'build-dependencies.txt'
    dependencies.write_text('\n'.join(line for line in dependencies.read_text(encoding='utf-8-sig').splitlines() if line.split('==')[0].lower() not in versions)+'\n'+''.join(k+'=='+v+'\n' for k,v in versions.items()),encoding='utf-8')
    files={p.relative_to(app).as_posix():digest(p) for p in app.rglob('*') if p.is_file()}
    manifest=dict(version=version,build='audited CPU runtime extension',baseline_runtime_sha256=args.sha256,
        app_modules=app_files,dependency_modules=dependency_files,dependency_versions=versions,
        entry_sha256=digest(source/'release_tools/entry.py'),files=files)
    (app/'source-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(app=str(app),version=version,files=len(files),app_modules=len(app_files),dependency_modules=len(dependency_files),dependency_versions=versions),ensure_ascii=False))


if __name__=='__main__':main()
