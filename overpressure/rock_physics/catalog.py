"""Read source signatures without importing optional scientific dependencies."""
import ast
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import re
from .presets import EXAMPLES,TITLES,OUTPUT_NAMES

VENDOR=Path(__file__).resolve().parents[1]/'vendor'/'rockphypy'
CLASSES={
    'utils':('utils','弹性参数与刚度矩阵'), 'EM':('EM','有效介质与孔隙裂缝'),
    'GM':('GM','颗粒接触与胶结'), 'Fluid':('Fluid','流体替换与频散'),
    'BW':('BW','温压流体性质'), 'Empirical':('Emp','经验关系'),
    'Anisotropy':('Anisotropy','各向异性'), 'AVO':('AVO','AVO 与反射透射'),
    'Permeability':('Perm','渗透率'), 'QI':('QI','定量解释辅助'),
}
INTERNAL={'EM.DEM','EM.OC_R_funcs','EM.Berryman_func','BW.dz_dp'}


def parameter_docs(doc):
    section=doc.split('Parameters',1)[-1].split('Returns',1)[0]
    result={};current=[]
    for line in section.splitlines():
        match=re.match(r'^\s*([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*:\s*(.+)',line)
        if match:
            current=[name.strip() for name in match[1].split(',')]
            for name in current:result[name]=dict(type=match[2],description='')
        elif current and line.strip() and not set(line.strip())<=set('-'):
            for name in current:result[name]['description']+=((' ' if result[name]['description'] else '')+line.strip())
    return result


def unit_for(model_id,name,description):
    if model_id.startswith('utils.'):
        if name in ('rho','den'):return 'g/cc'
        if name in ('vp','vs'):return 'm/s'
    if model_id.startswith('BW.'):
        return {'T':'°C','P':'MPa','S':'质量分数（ppm / 1e6）','den':'g/cc','Rg':'L/L'}.get(name,'')
    if model_id.startswith('GM.') and name=='sigma':return 'MPa'
    if model_id=='Anisotropy.Thomsen' and name=='den':return 'g/cc'
    if model_id.startswith(('utils.','EM.','GM.','Fluid.')) and name.startswith(('K','G')) and name not in ('Gamma',):return 'GPa'
    if model_id.startswith('Fluid.') and name in ('rho0','rho1','rhofl','rhofl1','rhofl2','D0','Df'):return 'g/cc'
    if model_id=='Fluid.Gassmann_vels' and name in ('Vp1','Vs1'):return 'm/s'
    if model_id.startswith('QI.') and name.startswith('M_'):return 'GPa'
    if model_id.startswith('QI.') and name.startswith('den_'):return 'g/cc'
    if model_id=='Permeability.Kozeny_Carman' and name=='d':return 'm'
    if name.lower() in ('theta','azimuth'):return '°'
    if name.lower().startswith('phi') or name in ('Sw','Sco2','vsh','volumes'):return '体积分数 0–1'
    for pattern in ('GPa','MPa','m/s','kg/m3','g/cm3','g/cc','mD','degree'):
        if pattern in description:return pattern
    return ''


@lru_cache(maxsize=1)
def _catalog():
    entries=[]
    for class_name,(module,category) in CLASSES.items():
        source=VENDOR/(module+'.py');tree=ast.parse(source.read_text(encoding='utf-8'))
        cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name==class_name)
        for function in cls.body:
            if not isinstance(function,ast.FunctionDef) or function.name.startswith('_'):continue
            names=[arg.arg for arg in function.args.args]
            model_id=class_name+'.'+function.name
            if 'self' in names or 'cls' in names or 'plot' in function.name.lower() or model_id in INTERNAL:continue
            if function.args.vararg or function.args.kwarg:continue
            doc=ast.get_docstring(function) or '';docs=parameter_docs(doc);parameters=[]
            defaults=[None]*(len(names)-len(function.args.defaults))+list(function.args.defaults)
            for name,default in zip(names,defaults):
                item=docs.get(name,docs.get('Cij',{}))
                try:default_value=ast.literal_eval(default) if default is not None else None
                except (ValueError,TypeError):default_value=None
                if model_id in EXAMPLES and name in EXAMPLES[model_id]:default_value=deepcopy(EXAMPLES[model_id][name])
                description=item.get('description','')
                parameters.append(dict(name=name,default=default_value,required=default is None,
                    description=description,type=item.get('type','number or array'),unit=unit_for(model_id,name,description)))
            returns=[node for node in ast.walk(function) if isinstance(node,ast.Return)]
            output_names=OUTPUT_NAMES.get(model_id)
            if not output_names and returns:
                value=returns[-1].value
                variables=value.elts if isinstance(value,(ast.Tuple,ast.List)) else [value]
                output_names=[v.id if isinstance(v,ast.Name) else 'output_'+str(i+1) for i,v in enumerate(variables)]
            entries.append(dict(id=model_id,category=category,title=TITLES.get(model_id,function.name),
                description=doc,parameters=parameters,example=deepcopy(EXAMPLES.get(model_id,{})),
                validated_example=model_id in EXAMPLES,output_names=output_names or ['result'],
                source=dict(file=module+'.py',line=function.lineno),library='RockPhyPy 0.0.2',license='LGPL-3.0'))
    return sorted(entries,key=lambda item:(list(CLASSES).index(item['id'].split('.')[0]),not item['validated_example'],item['id']))


def catalog():
    """Return editable JSON-safe metadata, independent of import-time side effects."""
    return deepcopy(_catalog())


def model_spec(model_id):
    entry=next((entry for entry in _catalog() if entry['id']==model_id),None)
    if entry is None:raise ValueError('未知岩石物理模型：'+str(model_id))
    return entry
