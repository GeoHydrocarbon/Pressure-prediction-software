"""Reproducible model parameters and tabular outputs in isolated result folders."""
import csv
from datetime import datetime
import json
from pathlib import Path


def save_result(result,output_dir):
    if result.get('preview') is not None:raise ValueError('实时预览不能作为完整成果保存，请运行完整计算')
    if result.get('status')!='completed':raise ValueError('只能保存完成的岩石物理结果')
    root=Path(output_dir).resolve();root.mkdir(parents=True,exist_ok=True)
    suffix=result['model_id'].replace('.','_')
    path=root/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+suffix)
    path.mkdir(exist_ok=False)
    (path/'parameters.json').write_text(json.dumps(dict(model_id=result['model_id'],parameters=result['parameters']),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    with (path/'results.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(result['table']['columns']);writer.writerows(result['table']['rows'])
    (path/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return path
