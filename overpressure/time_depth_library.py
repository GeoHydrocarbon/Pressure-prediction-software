"""Immutable named time/depth configurations reused by individual wells."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from .config import Project
from .time_depth import TimeDepth


def relation(project,config):
    cfg={key:deepcopy(config[key]) for key in ('method','time_unit','time_convention','depth_unit')}
    if cfg['method']=='table':cfg['table_file']=str(project.resolve(config['table_file']))
    else:cfg['coefficients']=list(config['coefficients'])
    TimeDepth(Project(project.path,{'time_depth':cfg}))
    ident=hashlib.sha256(json.dumps(cfg,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:24]
    name=('表：'+Path(cfg['table_file']).name) if cfg['method']=='table' else '公式：'+', '.join(map(str,cfg['coefficients']))
    name+=f" [{cfg['time_unit']}, {cfg['time_convention']}, {cfg['depth_unit']}]"
    return dict(id=ident,name=name,config=cfg)
