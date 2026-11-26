"""Register selected derived logs without overwriting existing well curves."""
from copy import deepcopy
from pathlib import Path

from .well_model import OUTPUT_UNITS


def attach_curves(wells,path,result,selection):
    if result.get('preview') is not None:raise ValueError('实时预览不能写入井曲线，请先运行完整计算')
    if result.get('kind')!='well_log':raise ValueError('不是测井建模成果')
    updated=deepcopy(wells);well=next((w for w in updated if w['name']==result['well']),None)
    if well is None:raise ValueError('项目中已不存在成果对应的井')
    names={c['name'] for c in well['curves']};selected=[]
    columns=result['table']['columns'];rows=result['table']['rows']
    for column,name in selection.items():
        name=name.strip()
        if column not in OUTPUT_UNITS or column not in columns:raise ValueError('不支持的成果曲线：'+column)
        if not name or name in names or name in ('Depth','well','source_row','source_file','source_file_row'):raise ValueError('曲线名称为空、重复或使用保留名称：'+name)
        index=columns.index(column)
        if not any(row[index] is not None for row in rows):raise ValueError(column+' 没有有效成果值')
        names.add(name);selected.append((column,name))
    if not selected:raise ValueError('至少选择一条成果曲线')
    file=Path(path).resolve()/'results.csv'
    if not file.is_file():raise ValueError('成果数据文件不存在')
    z=[row[columns.index('Depth')] for row in rows]
    for column,name in selected:
        well['curves'].append(dict(name=name,column=column,unit=OUTPUT_UNITS[column],path=str(file),
            well_column='',well_value=well['name'],depth_column='Depth',depth_unit='m',
            depth_min=min(z),depth_max=max(z),rows=len(z),role='derived',method='rock_physics',
            scheme_name=result['scheme_name'],provenance_sha256=result['provenance_sha256'],result_path=str(Path(path).resolve())))
    return updated
