"""Rebuild application bytecode using an integrity-checked local frozen CPU runtime.

This does not change the third-party runtime or resolve new dependencies. Frozen
smoke checks are mandatory before packaging. Archive structures are documented in
https://github.com/pyinstaller/pyinstaller/tree/v6.22.0/PyInstaller/archive .
"""
import argparse
import hashlib
import importlib.util
import io
import json
import marshal
from pathlib import Path
import struct
import sys
import zipfile
import zlib

COOKIE=struct.Struct('!8sIIII64s')
ENTRY=struct.Struct('!IIIIBc')
MAGIC=b'MEI\x0c\x0b\x0a\x0b\x0e'


def digest(path):
    with path.open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def unpack_executable(path):
    blob=path.read_bytes();cookie_pos=blob.rfind(MAGIC)
    if cookie_pos<0:raise ValueError('No frozen runtime archive')
    magic,length,tocpos,tocsize,version,library=COOKIE.unpack_from(blob,cookie_pos)
    if cookie_pos+COOKIE.size!=len(blob):raise ValueError('Trailing/signed executable is unsupported')
    if version!=sys.version_info.major*100+sys.version_info.minor:raise ValueError('Python minor version mismatch')
    start=cookie_pos+COOKIE.size-length;toc=blob[start+tocpos:start+tocpos+tocsize]
    entries=[];cursor=0
    while cursor<len(toc):
        size,offset,stored,original,compressed,kind=ENTRY.unpack_from(toc,cursor)
        if size<ENTRY.size or cursor+size>len(toc):raise ValueError('Invalid archive table')
        name=toc[cursor+ENTRY.size:cursor+size].rstrip(b'\0').decode('utf-8')
        entries.append(dict(name=name,kind=kind,data=blob[start+offset:start+offset+stored],
                            compressed=compressed,original=original))
        cursor+=size
    return blob[:start],entries,library


def payload(entry):
    return zlib.decompress(entry['data']) if entry['compressed'] else entry['data']


def rebuild_pyz(blob,source):
    if blob[:4]!=b'PYZ\0' or blob[4:8]!=importlib.util.MAGIC_NUMBER:raise ValueError('Frozen bytecode version mismatch')
    toc=dict(marshal.loads(blob[struct.unpack('!i',blob[8:12])[0]:]))
    # Retain the byte-for-byte compiled dependencies, replace every app module.
    modules={name:(kind,blob[offset:offset+size]) for name,(kind,offset,size) in toc.items()
             if name!='overpressure' and not name.startswith('overpressure.')}
    files={}
    for file in sorted((source/'overpressure').rglob('*.py')):
        relative=file.relative_to(source)
        name='.'.join(relative.with_suffix('').parts)
        package=file.name=='__init__.py'
        if package:name=name.removesuffix('.__init__')
        code=compile(file.read_bytes(),relative.as_posix(),'exec',dont_inherit=True)
        modules[name]=(int(package),zlib.compress(marshal.dumps(code),6))
        files[relative.as_posix()]=digest(file)
    output=io.BytesIO();output.write(b'\0'*17);new_toc=[]
    for name,(kind,data) in sorted(modules.items()):
        new_toc.append((name,(kind,output.tell(),len(data))));output.write(data)
    offset=output.tell();output.write(marshal.dumps(new_toc));output.seek(0)
    output.write(b'PYZ\0'+importlib.util.MAGIC_NUMBER+struct.pack('!i',offset))
    return output.getvalue(),files


def rewrite_executable(path,source):
    bootloader,entries,library=unpack_executable(path)
    replaced=[];files={}
    for entry in entries:
        if entry['kind']==b'z':
            data,files=rebuild_pyz(payload(entry),source);replaced.append('PYZ')
        elif entry['kind']==b's' and entry['name']=='entry':
            data=marshal.dumps(compile((source/'release_tools'/'entry.py').read_bytes(),
                                      'release_tools/entry.py','exec',dont_inherit=True));replaced.append('entry')
        else:continue
        entry.update(data=zlib.compress(data,9) if entry['compressed'] else data,original=len(data))
    if sorted(replaced)!=['PYZ','entry']:raise ValueError(f'Unexpected archive layout: {replaced}')
    data=io.BytesIO();toc=[]
    for entry in entries:
        offset=data.tell();data.write(entry['data'])
        name=entry['name'].encode('utf-8')+b'\0'
        length=(ENTRY.size+len(name)+15)//16*16
        toc.append(ENTRY.pack(length,offset,len(entry['data']),entry['original'],entry['compressed'],entry['kind'])+
                   name.ljust(length-ENTRY.size,b'\0'))
    offset=data.tell();table=b''.join(toc);data.write(table)
    data.write(COOKIE.pack(MAGIC,data.tell()+COOKIE.size,offset,len(table),
                          sys.version_info.major*100+sys.version_info.minor,library))
    path.write_bytes(bootloader+data.getvalue())
    # Read it back rather than assuming the regenerated table is well formed.
    _,check,_=unpack_executable(path)
    for entry in check:
        if len(payload(entry))!=entry['original']:raise ValueError('Archive length verification failed')
    return files


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--runtime-sha256',required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--refresh',action='store_true',help='Refresh only a previously rebuilt staging directory')
    args=parser.parse_args()
    source=args.source.resolve();output=args.output.resolve()
    if (source/'overpressure'/'rock_physics').is_dir():
        raise ValueError('This source needs RockPhyPy source/license resources. Use the full windows.spec build; the old offline runtime baseline does not contain these resources.')
    workspace=Path(__file__).resolve().parents[1]
    if not output.is_relative_to(workspace/'outputs'):raise ValueError('Output must stay within workspace outputs')
    actual=digest(args.runtime)
    if actual!=args.runtime_sha256:raise ValueError('Baseline runtime SHA256 mismatch')
    app=output/'PressurePrediction'
    if args.refresh:
        previous=json.loads((app/'source-manifest.json').read_text(encoding='utf-8'))
        if previous['baseline_runtime_sha256']!=actual:raise ValueError('Existing staging runtime does not match')
    else:
        if output.exists():raise FileExistsError(output)
        output.mkdir(parents=True)
        with zipfile.ZipFile(args.runtime) as archive:
            for entry in archive.infolist():
                destination=(output/entry.filename).resolve()
                if not destination.is_relative_to(output):raise ValueError('Unsafe runtime archive path')
            archive.extractall(output)
    manifests={}
    for name in ('PressurePrediction.exe','PressureWorker.exe'):
        manifests[name]=rewrite_executable(app/name,source)
    if manifests['PressurePrediction.exe']!=manifests['PressureWorker.exe']:raise ValueError('GUI and worker modules differ')
    import shutil
    shutil.copyfile(source/'project.example.yaml',app/'_internal'/'project.example.yaml')
    shutil.copyfile(source/'release_tools'/'README-windows.txt',app/'README.txt')
    version={};exec((source/'overpressure'/'__init__.py').read_text(encoding='utf-8'),version)
    (app/'build-info.txt').write_text(f"version={version['__version__']}\nruntime=Windows x64 CPU\nbuild=offline application rebuild\n",encoding='utf-8')
    manifest=dict(version=version['__version__'],build='offline application rebuild',baseline_runtime_sha256=actual,
                  app_modules=manifests['PressurePrediction.exe'],template_sha256=digest(source/'project.example.yaml'),
                  entry_sha256=digest(source/'release_tools'/'entry.py'),
                  executable_sha256={name:digest(app/name) for name in manifests})
    (app/'source-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(app=str(app),version=version['__version__'],modules=len(manifest['app_modules'])),ensure_ascii=False))


if __name__=='__main__':main()
