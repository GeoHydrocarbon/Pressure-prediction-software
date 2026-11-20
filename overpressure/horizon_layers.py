"""Small GUI-side registry; mapped arrays remain in the reader process."""
from pathlib import Path
import hashlib

COLORS=['#e35050','#267cb5','#d69a28','#8d64b2','#26a99a','#d665a5']


def add_layers(entries,summary):
    source=str(Path(summary['source']).resolve());previous={v['name']:v for v in entries.values() if v['source']==source}
    result={key:dict(value) for key,value in entries.items() if value['source']!=source}
    if len(result)+len(summary['names'])>64:raise ValueError('同时管理的层位最多 64 层，请先清除部分层位')
    for name in summary['names']:
        key=hashlib.sha256((source+'\0'+name).encode('utf-8')).hexdigest()[:24]
        old=previous.get(name,{})
        result[key]=dict(id=key,source=source,path=summary['path'],name=name,label=name,
                         derived=summary.get('derived',False),interpolation_report=summary.get('interpolation_report',''),
                         checked=old.get('checked',True),width=old.get('width',1.6),color=old.get('color',COLORS[len(result)%len(COLORS)]))
    counts={}
    for item in result.values():counts[item['name']]=counts.get(item['name'],0)+1
    for item in result.values():item['label']=item['name']+(f" [{Path(item['source']).name}]" if counts[item['name']]>1 else '')
    return result
