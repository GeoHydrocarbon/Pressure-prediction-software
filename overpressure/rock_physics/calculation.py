"""Validated adapter around replaceable RockPhyPy numerical source modules."""
from importlib import import_module
import inspect
import json
import math
import warnings
import hashlib
import numpy as np
from .catalog import CLASSES,model_spec,VENDOR
from .. import __version__

MAX_INPUT_VALUES=50000
MAX_OUTPUT_VALUES=200000


def numeric_value(value,name,budget,*,allow_bool=False,allow_text=False,allow_none=False):
    if value is None and allow_none:return None
    if value is None:raise ValueError(f'参数 {name} 未填写；必填物理参数不能为 null')
    if isinstance(value,bool):
        if not allow_bool:raise ValueError(f'参数 {name} 需要物理数值，不能使用 true / false')
        return value
    if isinstance(value,str):
        if not allow_text:raise ValueError(f'参数 {name} 需要数值，不能使用文本')
        if len(value)>200:raise ValueError(f'参数 {name} 文本过长')
        return value
    if isinstance(value,(int,float)):
        if not math.isfinite(value):raise ValueError(f'参数 {name} 必须为有限数值')
        budget[0]+=1;return value
    if isinstance(value,(list,tuple,np.ndarray)):
        def has_bool(items):
            return any(has_bool(item) if isinstance(item,(list,tuple)) else isinstance(item,(bool,np.bool_)) for item in items)
        if isinstance(value,np.ndarray) and value.dtype.kind=='b' or not isinstance(value,np.ndarray) and has_bool(value):raise ValueError(f'参数 {name} 的数值数组不能包含 true / false')
        try:array=np.asarray(value,dtype=float)
        except (TypeError,ValueError) as exc:raise ValueError(f'参数 {name} 需要规则的数值数组，不能包含文本或不等长行') from exc
        if array.ndim>3 or not array.size or not np.isfinite(array).all():raise ValueError(f'参数 {name} 需要 1～3 维、非空且有限的数组')
        budget[0]+=array.size;return array
    raise ValueError(f'参数 {name} 仅支持数值、数值数组、布尔或模型定义的选项文本')


def _all(value,condition):
    return bool(np.all(condition(np.asarray(value))))


def validate_domain(model_id,p):
    shapes=[value.shape for value in p.values() if isinstance(value,np.ndarray)]
    if shapes:
        try:elements=math.prod(np.broadcast_shapes(*shapes))
        except ValueError:elements=0  # Some models deliberately use independent axes/matrix products.
        if elements>MAX_OUTPUT_VALUES:raise ValueError('数组广播后的规模过大，请减少扫描点数或使用一维数组')
    if model_id=='AVO.Aki_Richards':
        size=np.size(p['vp1'])*np.size(p['theta']) if np.ndim(p['vp1']) else np.size(p['theta'])
        if size>MAX_OUTPUT_VALUES//4:raise ValueError('速度样点与角度组成的反射网格过大，请减少点数')
    if model_id=='AVO.AVO_HTI' and np.size(p['theta'])*np.size(p['azimuth'])>MAX_OUTPUT_VALUES//2:raise ValueError('入射角与方位角网格过大，请减少点数')
    for key,value in p.items():
        if value is None or isinstance(value,(str,bool)):continue
        if key.lower().startswith('phi') or key in ('Sw','Sco2','vsh','volumes'):
            if not _all(value,lambda x:(x>=0)&(x<=1)):raise ValueError(f'{key} 必须为 0～1 的分数，不能直接填百分数')
        if key in ('K','K0','K1','K2','K_mat','Kdry','K_dry','Kfl','Kf','Km','Ki','Ks','Kg','Kc','G','G0','G1','G2','Gdry','G_dry','Gm','Gi','Gs','Gg','Gc'):
            if not _all(value,lambda x:x>=0):raise ValueError(f'{key} 不能为负数')
        if key in ('rho','rho0','rhofl','rhofl1','rhofl2','rho1','rho2','den1','den2','D0','Df','Dqz','Dsh','Dc','Db','Dh','Dhc') or key=='den' and not isinstance(value,bool):
            if not _all(value,lambda x:x>0):raise ValueError(f'{key} 密度必须大于 0，请核对模型要求的单位')
        if key=='C' and model_id=='Empirical.han' and not _all(value,lambda x:(x>=0)&(x<=1)):raise ValueError('黏土体积分数 C 必须在 0～1 之间')
        if key in ('Cn','d','a','kapa','eta','freq'):
            if not _all(value,lambda x:x>0):raise ValueError(f'{key} 必须大于 0')
        if key=='iter_n' and (not isinstance(value,int) or not 1<=value<=10000):raise ValueError('迭代次数必须为 1～10000 的整数')
    for key in ('bound','mode','scheme','axis','order'):
        if key not in p:continue
        choices=None
        if key=='bound':choices=('upper','lower')
        if key=='axis':choices=(1,3) if model_id in ('Anisotropy.Bond_trans','EM.hudson') else (1,2,3)
        if key=='order':choices=(1,2)
        if key=='scheme' and model_id.startswith('GM.'):choices=(1,2)
        if choices is not None and p[key] not in choices:raise ValueError(f'{key} 必须从 {choices} 中选择')
    if model_id=='EM.VRH':
        volume=np.asarray(p['volumes']);moduli=np.asarray(p['M'])
        if moduli.ndim!=1 or volume.ndim not in (1,2) or volume.shape[-1]!=len(moduli):raise ValueError('VRH 的各相模量 M 与体积分数最后一维必须一一对应')
        if not np.allclose(volume.sum(axis=-1),1.,atol=1e-6,rtol=0):raise ValueError('每组矿物体积分数之和必须为 1，不自动归一化')
        if not np.all(moduli>0):raise ValueError('VRH 各相模量必须大于 0，含零剪切模量的流体请使用合适的混合模型')
    if model_id in ('EM.cripor','GM.softsand','GM.stiffsand','GM.contactcement'):
        if not _all(p['phic'],lambda x:(x>0)&(x<1)):raise ValueError('临界孔隙度 phic 必须在 0 与 1 之间')
        if np.any(np.asarray(p['phi'])>np.asarray(p['phic'])):raise ValueError('此模型要求孔隙度不大于临界孔隙度')
    if model_id=='EM.Berryman_DEM':
        for name in ('Km','Gm','Ki','Gi','alpha','phi'):
            if np.ndim(p[name])!=0:raise ValueError('Berryman DEM 的输入应为标量；返回从 0 到目标孔隙度的路径')
        if not 0<p['alpha']<=1:raise ValueError('DEM 孔隙纵横比 alpha 必须满足 0 < alpha <= 1')
        if not 0<=p['phi']<1:raise ValueError('DEM 孔隙度必须满足 0 <= phi < 1')
        if p['Km']<=0 or p['Gm']<=0:raise ValueError('DEM 基质体积和剪切模量必须大于 0')
    if model_id.startswith('Fluid.Gassmann') or model_id=='Fluid.vels':
        if not _all(p['phi'],lambda x:(x>0)&(x<1)):raise ValueError('Gassmann 孔隙度必须满足 0 < phi < 1')
        for key in ('K_mat','K0','Kf','Kfl1','Kfl2'):
            if key in p and not _all(p[key],lambda x:x>0):raise ValueError(f'{key} 必须大于 0')
        if model_id=='Fluid.Gassmann' and np.any(np.asarray(p['K_dry'])>np.asarray(p['K_mat'])):raise ValueError('干岩体积模量不能大于无孔基质模量')
    if model_id.startswith('BW.'):
        if 'T' in p and not _all(p['T'],lambda x:x>-273.15):raise ValueError('温度必须高于绝对零度（单位 °C）')
        if 'P' in p and not _all(p['P'],lambda x:x>0):raise ValueError('流体模型压力必须大于 0，单位 MPa')
        if 'S' in p and not _all(p['S'],lambda x:(x>=0)&(x<=1)):raise ValueError('盐度 S 使用质量分数，例如 35000 ppm 填 0.035')
    if model_id.startswith('AVO.'):
        for key in ('theta','the'):
            if key in p and not _all(p[key],lambda x:(x>=0)&(x<90)):raise ValueError('入射角必须在 0°（含）到 90°（不含）之间')
        if model_id=='AVO.zoeppritz':
            for name in ('vp1','vs1','vp2','vs2'):
                if not _all(p[name],lambda x:x>0):raise ValueError('本 Zoeppritz 模型需要大于 0 的纵横波速度')
            maximum=max(p['vp2'],p['vs1'],p['vs2'])
            if maximum>p['vp1']:
                critical=np.rad2deg(np.arcsin(p['vp1']/maximum))
                if np.any(np.asarray(p['theta'])>=critical):raise ValueError(f'入射角达到临界角 {critical:.4g}°；本模型仅返回亚临界的实数反射系数')
    if model_id=='Anisotropy.Bond_trans':
        if p.get('axis',3) not in (1,3):raise ValueError('此刚度旋转模型支持 axis=1 或 axis=3')
        if np.shape(p['C'])!=(6,6) or not np.allclose(p['C'],np.asarray(p['C']).T,atol=1e-8):raise ValueError('刚度 C 必须为对称的 6×6 矩阵')


def output_unit(model_id,name):
    if model_id in ('Fluid.Biot','Fluid.Biot_HF','Fluid.Geertsma_Smit_HF') and name.startswith(('Vp','Vs')):return 'km/s'
    if model_id=='Fluid.Geertsma_Smit_LF':return '与输入速度一致'
    if model_id=='Permeability.Kozeny_Carman':return 'm²'
    if name.startswith(('Vp','Vs','VP','VS')) or model_id in ('BW.v_water','BW.v_brine'):return 'm/s'
    if name.startswith('density'):return 'g/cc'
    if name in ('K','G','K_dry','G_dry','K_sat','G_sat','K_sat_2','lambda','Voigt','Reuss','Hill') or name.startswith('C') and model_id.startswith(('utils.','Anisotropy.')):return 'GPa'
    if name=='porosity':return 'fraction'
    return ''


def table_from_outputs(outputs,parameters):
    arrays=[np.asarray(item['values']) for item in outputs]
    if all(array.ndim<=1 for array in arrays):
        count=max((array.size for array in arrays),default=1)
        if all(array.size in (1,count) for array in arrays):
            vectors=[np.repeat(array.item(),count) if array.size==1 else array for array in arrays]
            axis=next(((name,np.asarray(value)) for name,value in parameters.items() if isinstance(value,list) and np.asarray(value).ndim==1 and len(value)==count and count>1),None)
            columns=['input.'+axis[0] if axis else 'index']+[item['name']+(' ['+item['unit']+']' if item['unit'] else '') for item in outputs]
            rows=np.column_stack([axis[1] if axis else np.arange(count),*vectors]).tolist()
            return dict(columns=columns,rows=rows)
    dimension=max((array.ndim for array in arrays),default=1)
    columns=['output',*['i'+str(i+1) for i in range(dimension)],'value'];rows=[]
    for item,array in zip(outputs,arrays):
        if array.ndim==0:rows.append([item['name'],*([None]*dimension),array.item()]);continue
        for index in np.ndindex(array.shape):rows.append([item['name'],*index,*([None]*(dimension-len(index))),array[index].item()])
    return dict(columns=columns,rows=rows)


def calculate(model_id,parameters):
    spec=model_spec(model_id)
    if not isinstance(parameters,dict):raise ValueError('模型参数必须为名称与数值的映射')
    names={p['name'] for p in spec['parameters']}
    if set(parameters)-names:raise ValueError('未知参数：'+', '.join(sorted(set(parameters)-names)))
    supplied={};budget=[0]
    for item in spec['parameters']:
        name=item['name']
        if name in parameters:
            supplied[name]=numeric_value(parameters[name],name,budget,
                allow_bool='bool' in item['type'].lower() or isinstance(item['default'],bool),
                allow_text=any(word in item['type'].lower() for word in ('str','string')) or isinstance(item['default'],str),
                allow_none=not item['required'] and item['default'] is None)
        elif item['required']:raise ValueError('缺少必填参数：'+name)
        # Omitted optional parameters retain the upstream implementation's default.
    if budget[0]>MAX_INPUT_VALUES:raise ValueError(f'单次输入最多 {MAX_INPUT_VALUES:,} 个数值，请缩小扫描范围')
    class_name,method=model_id.split('.');module_name=CLASSES[class_name][0]
    module=import_module('overpressure.vendor.rockphypy.'+module_name);function=getattr(getattr(module,class_name),method)
    for name,param in inspect.signature(function).parameters.items():
        if name not in supplied and param.default is not inspect.Parameter.empty:supplied[name]=param.default
    validate_domain(model_id,supplied)
    try:
        with warnings.catch_warnings(record=True) as caught,np.errstate(all='warn'):
            warnings.simplefilter('always');value=function(**supplied)
        if any(type(w.message).__name__ in ('ODEintWarning','OptimizeWarning') for w in caught):raise ValueError('数值求解未收敛，请调整模型参数')
    except (ValueError,TypeError,ZeroDivisionError,OverflowError,IndexError,FloatingPointError,np.linalg.LinAlgError) as exc:
        raise ValueError(f"{spec['title']} 计算失败：{exc}。请核对参数单位、维度和模型适用条件") from exc
    values=value if isinstance(value,tuple) else (value,)
    outputs=[];size=0
    for i,item in enumerate(values):
        name=spec['output_names'][i] if i<len(spec['output_names']) else f'output_{i+1}'
        array=np.asarray(item)
        if array.dtype.kind not in 'biufc' or not array.size:raise ValueError('模型没有返回可显示的非空数值结果')
        size+=array.size
        if size>MAX_OUTPUT_VALUES:raise ValueError('结果超过显示上限，请缩小输入数组或扫描范围')
        if not np.isfinite(array).all():raise ValueError('模型结果包含非有限值；输入可能超出适用范围，请核对孔隙度、模量、频率和角度')
        components=[(name+'.real',array.real),(name+'.imag',array.imag)] if np.iscomplexobj(array) else [(name,array)]
        for label,component in components:outputs.append(dict(name=label,values=component.tolist(),shape=list(component.shape),unit=output_unit(model_id,name)))
    serial_parameters={name:value.tolist() if isinstance(value,np.ndarray) else value for name,value in supplied.items()}
    notes=list(dict.fromkeys(str(w.message) for w in caught))
    if np.iscomplexobj(np.asarray(values[0])):notes.append('复数结果以实部与虚部分列保存')
    result=dict(schema_version=1,status='completed',model_id=model_id,title=spec['title'],category=spec['category'],
        software_version=__version__,
        parameters=serial_parameters,outputs=outputs,table=table_from_outputs(outputs,serial_parameters),warnings=notes,
        library=dict(name='RockPhyPy',version='0.0.2',license='LGPL-3.0',source_commit='6ad244cb95e01e1c2491b38ad871605b9db5fb8d',
                     source_manifest_sha256=hashlib.sha256((VENDOR/'SOURCE.json').read_bytes()).hexdigest()),
        model_description=spec['description'])
    json.dumps(result,allow_nan=False)
    return result
