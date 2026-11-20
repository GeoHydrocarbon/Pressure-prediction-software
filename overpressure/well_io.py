"""Read selected well files with stable provenance and per-file column inference."""
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

ALIASES={'well':['井','井名','井号','well','wellname','wellid'],'Depth':['Depth','深度','井深','测点深度','测深','MD','DEPT','TVD'],
         'AC':['AC','DT','声波时差'],'DEN':['DEN','RHOB','密度'],'Ip':['Ip','纵波阻抗','波阻抗'],
         'Pp':['Pp','压力','实测压力','孔隙压力','pressure']}


def infer_columns(headers,current):
    normalize=lambda value:re.sub(r'[\s_\-]+','',str(value)).lower()
    result=dict(current)
    for field in current:
        if current[field] in headers:continue
        matches=[str(h) for h in headers if normalize(h) in {normalize(a) for a in ALIASES.get(field,[field])}]
        if len(matches)==1:result[field]=matches[0]
    return result


def selected_paths(project,config,*,measured=False):
    key='measurement_files' if measured else 'well_files'
    # An explicit empty list means no selected files; never silently restore old input.
    values=config[key] if key in config else [config.get('measurement_file' if measured else 'well_file','')]
    paths=[project.resolve(value) for value in values if value]
    if len(paths)!=len(set(paths)):raise ValueError('同一个文件不能重复导入')
    return paths


def read_well_files(paths,columns,*,required,allow_missing_well=True,reject_duplicate_wells=True):
    from .data import read_table,sha256_file
    required=list(dict.fromkeys(required))
    if not paths:raise ValueError('请至少选择一个井数据文件')
    frames=[];sources=[];owners={}
    for path in paths:
        path=Path(path)
        headers=read_table(path,nrows=0).columns;mapping=infer_columns(headers,columns)
        raw=read_table(path,dtype={mapping['well']:'string'} if mapping['well'] in headers else None)
        if raw.empty:raise ValueError(f'{path.name}：没有数据行')
        missing={mapping[k] for k in required if k!='well'}-set(raw.columns)
        if missing:raise ValueError(f'{path.name}：缺少列 {sorted(missing)}')
        if len({mapping[k] for k in required})!=len(required):raise ValueError(f'{path.name}：字段对应不能重复')
        frame=pd.DataFrame(index=raw.index)
        for key,target in columns.items():
            if mapping[key] in raw:frame[target]=raw[mapping[key]]
            elif key=='well' and allow_missing_well:frame[target]=path.stem
            elif key in required:raise ValueError(f'{path.name}：缺少 {key}')
        names=frame[columns['well']].astype('string').str.strip();frame[columns['well']]=names
        for name in names.dropna().unique():
            if not name:continue
            if reject_duplicate_wells and name in owners:raise ValueError(f'井 {name} 同时出现在 {owners[name]} 与 {path.name}，请取消重复井文件')
            owners[name]=path.name
        frame['_source_file']=str(path);frame['_source_file_row']=np.arange(len(frame))+2
        frames.append(frame);sources.append(dict(path=str(path),sha256=sha256_file(path),rows=len(frame),wells=names.dropna().unique().tolist(),columns=mapping))
    combined=pd.concat(frames,ignore_index=True)
    digest=sources[0]['sha256'] if len(sources)==1 else hashlib.sha256(json.dumps(sources,ensure_ascii=False,sort_keys=True).encode('utf-8')).hexdigest()
    return combined,sources,digest
