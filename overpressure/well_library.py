"""Well-centric curve references and depth-aligned, canonical training data."""
from copy import deepcopy
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import UNITS,CANONICAL_UNITS,RESERVED_FEATURES,unit_factor,feature_units


def curve_key(column):
    from .well_io import ALIASES
    normalized=str(column).strip().lower()
    for name,aliases in ALIASES.items():
        if name!='well' and normalized in {a.lower() for a in aliases}:return name
    return str(column).strip()


def inspect_curve_file(path,*,well_column='',depth_column='Depth',depth_unit='m',forced_well='',definitions=None,default_units=None,logical_names=None):
    from .data import read_table
    path=Path(path).resolve();raw=read_table(path,dtype={well_column:'string'} if well_column else None)
    if raw.empty:raise ValueError(f'{path.name}：文件没有数据')
    if depth_column not in raw:raise ValueError(f'{path.name}：缺少深度列 {depth_column}')
    names=raw[well_column].astype('string').str.strip() if well_column else pd.Series(forced_well or path.stem,index=raw.index,dtype='string')
    if forced_well:
        if well_column and set(names.dropna())!={forced_well}:raise ValueError(f'{path.name} 的井名与所选井 {forced_well} 不符；多井文件请在“井”根节点导入')
        names[:]=forced_well
    if names.isna().any() or names.eq('').any():raise ValueError(f'{path.name}：存在空井名，请先补全')
    if definitions is None:
        definitions=[]
        for column in raw:
            if column in (well_column,depth_column) or not pd.to_numeric(raw[column],errors='coerce').notna().any():continue
            key=(logical_names or {}).get(column,curve_key(column));definitions.append(dict(column=str(column),name=key,unit=(default_units or CANONICAL_UNITS).get(key,'')))
    if any(d['column'] not in raw for d in definitions):raise ValueError('曲线列不存在，请重新选择')
    keys=[d['name'] for d in definitions]
    if len(set(keys))!=len(keys) or any(not isinstance(k,str) or not k.strip() or k in (RESERVED_FEATURES-{'Pp'})|{'Depth'} for k in keys):raise ValueError('曲线名称不能为空、重复或使用系统保留名称')
    base=dict(path=str(path),well_column=well_column,depth_column=depth_column,depth_unit=depth_unit)
    wells=[]
    for name,group in raw.groupby(names,sort=False):
        depth=pd.to_numeric(group[depth_column],errors='coerce')*unit_factor('Depth',depth_unit)
        if not np.isfinite(depth).all() or (depth<0).any():raise ValueError(f'井 {name} 的深度含无效值或负值')
        if depth.round(6).duplicated().any():raise ValueError(f'井 {name} 存在重复深度，不能确定曲线对应关系')
        ref=dict(base,well_value=str(name),rows=len(group),depth_min=float(depth.min()),depth_max=float(depth.max()))
        curves=[dict(ref,name='Depth',column=depth_column,unit=depth_unit)]
        curves.extend(dict(ref,**definition) for definition in definitions)
        wells.append(dict(name=str(name),curves=curves))
    return wells


def merge_wells(existing,incoming):
    result=deepcopy(existing);by_name={w['name']:w for w in result}
    for item in incoming:
        if item['name'] not in by_name:
            well=dict(name=item['name'],curves=[]);result.append(well);by_name[item['name']]=well
        well=by_name[item['name']];curves={c['name']:c for c in well['curves']}
        curves.update({c['name']:deepcopy(c) for c in item['curves'] if c['name']!='Depth' or 'Depth' not in curves});well['curves']=list(curves.values())
    return result


def read_library_frame(project,names,features,target='Pp',*,require_labels=True,expected_units=None):
    """Use exact depth matches (1 µm rounding), without implicit interpolation."""
    from .data import read_table,sha256_file
    wells={w['name']:w for w in project.settings.get('project_data',{}).get('wells',[])}
    if not names:raise ValueError('请在训练设置中至少选择一口井')
    tables={};sources={};frames=[];units=expected_units or feature_units(project.settings,features)
    def read_curve(ref,name,as_target=False):
        path=project.resolve(ref['path']);key=str(path)
        if key not in tables:
            tables[key]=read_table(path,dtype={ref['well_column']:'string'} if ref.get('well_column') else None)
            sources[key]=dict(path=key,sha256=sha256_file(path),rows=len(tables[key]))
        raw=tables[key];rows=np.arange(len(raw))+2
        if ref.get('well_column'):
            mask=raw[ref['well_column']].astype('string').str.strip().eq(ref['well_value']).fillna(False).to_numpy()
            rows=rows[mask];raw=raw.loc[mask]
        depth=pd.to_numeric(raw[ref['depth_column']],errors='coerce').to_numpy(dtype=float)*unit_factor('Depth',ref['depth_unit'])
        if not np.isfinite(depth).all() or (depth<0).any() or pd.Series(depth.round(6)).duplicated().any():raise ValueError(f'{ref["well_value"]}：曲线 {name} 的深度无效或重复')
        values=depth if name=='Depth' and not as_target else pd.to_numeric(raw[ref['column']],errors='coerce').to_numpy(dtype=float)
        if as_target:values=values*unit_factor('Pp',ref['unit'])
        elif name in UNITS and name!='Depth':values=values*unit_factor(name,ref['unit'])
        elif name!='Depth' and ref.get('unit','')!=units.get(name,''):raise ValueError(f'{ref["well_value"]}：{name} 单位 {ref.get("unit", "")} 与模型/所选曲线单位 {units.get(name, "")} 不一致')
        return pd.Series(values,index=depth.round(6)),pd.Series(rows,index=depth.round(6)),key
    for name in names:
        if name not in wells:raise ValueError(f'井库中不存在井 {name}')
        curves={c['name']:c for c in wells[name]['curves']}
        needed=set(features)|{'Depth'}|({target} if require_labels else set())
        missing=needed-set(curves)
        if missing:raise ValueError(f'井 {name} 缺少曲线：{", ".join(sorted(missing))}；请导入曲线或取消选择该井')
        anchor=next((k for k in features if k!='Depth'),'Depth')
        values,rows,path=read_curve(curves[anchor],anchor)
        frame=pd.DataFrame(index=values.index);frame['Depth']=frame.index.to_numpy(dtype=float)
        frame['source_file']=path;frame['source_file_row']=rows
        for key in features:
            if key=='Depth':continue
            frame[key]=read_curve(curves[key],key)[0].reindex(frame.index)
        frame['Pp']=read_curve(curves[target],target,True)[0].reindex(frame.index) if target in curves else np.nan
        frame['well']=name;frames.append(frame.reset_index(drop=True))
    result=pd.concat(frames,ignore_index=True);result['source_row']=np.arange(len(result))+2
    provenance=list(sources.values());digest=hashlib.sha256(json.dumps(dict(sources=provenance,wells=[wells[n] for n in names],features=list(features),target=target),sort_keys=True,ensure_ascii=False).encode('utf-8')).hexdigest()
    return result,provenance,digest

def read_library_measurements(project,names):
    from .data import read_table,sha256_file
    tables={};frames=[];sources=[]
    for well in project.settings.get('project_data',{}).get('wells',[]):
        if well['name'] not in names:continue
        for ref in well['curves']:
            if ref.get('role')!='measurement':continue
            path=project.resolve(ref['path']);key=str(path)
            if key not in tables:
                tables[key]=read_table(path,dtype={ref['well_column']:'string'} if ref.get('well_column') else None)
                sources.append(dict(path=key,sha256=sha256_file(path)))
            raw=tables[key];rows=np.arange(len(raw))+2
            if ref.get('well_column'):
                mask=raw[ref['well_column']].astype('string').str.strip().eq(well['name']).fillna(False).to_numpy();raw=raw.loc[mask];rows=rows[mask]
            frames.append(pd.DataFrame(dict(well=well['name'],Depth=pd.to_numeric(raw[ref['depth_column']],errors='coerce').to_numpy()*unit_factor('Depth',ref['depth_unit']),Pp=pd.to_numeric(raw[ref['column']],errors='coerce').to_numpy()*unit_factor('Pp',ref['unit']),source_file=key,source_file_row=rows)))
    frame=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=['well','Depth','Pp','source_file','source_file_row'])
    frame['source_row']=np.arange(len(frame))+2
    frame.attrs['source_sha256']=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest();return frame
