"""Well-log baseline modelling using the vendored RockPhyPy kernels.

Inputs and observations have separate roles. Exact depth alignment is the default;
optional interpolation is restricted to neighbouring, finite samples within a
user-specified gap. Invalid samples remain in the result with QC reasons.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = {
    'phi': ('孔隙度', ['fraction', '%'], .1),
    'sw': ('含水饱和度', ['fraction', '%'], 1.),
    'toc': ('TOC（占干固体质量）', ['fraction', '%'], 0.),
    'temperature': ('温度', ['°C'], 80.),
    'pressure': ('地层压力', ['MPa', 'kPa', 'psi'], 30.),
    'vp': ('实测 Vp / 纵波时差', ['m/s', 'km/s', 'us/ft', 'us/m'], None),
    'vs': ('实测 Vs / 横波时差', ['m/s', 'km/s', 'us/ft', 'us/m'], None),
    'rho': ('实测密度', ['g/cm3', 'g/cc', 'kg/m3'], None),
}
PARAMETERS = {
    'soft_fraction': ('软孔占比', .15, 0., 1.),
    'alpha_hard': ('硬孔 / 单孔纵横比', .15, .001, 1.),
    'alpha_soft': ('软孔纵横比', .01, .001, 1.),
    'critical_porosity': ('临界孔隙度', .4, .001, .99),
}
OUTPUT_UNITS = dict(VP_RP='m/s', VS_RP='m/s', DEN_RP='g/cc', AC_RP='us/ft',
    IP_RP='g/cc*m/s', VPVS_RP='fraction', K_RP='GPa', G_RP='GPa')


def default_scheme():
    return dict(schema_version=1, name='基线方案1', well='', depth_min=0., depth_max=10000.,
        bindings={k: dict(curve='', constant=v[2], unit=v[1][0]) for k,v in FIELDS.items()},
        minerals=[dict(name='石英', K=37., G=44., density=2.65, curve='', constant=.7, unit='fraction'),
                  dict(name='黏土', K=21., G=7., density=2.60, curve='', constant=.3, unit='fraction')],
        mineral_basis='volume', normalize_minerals=True, kerogen=False,
        kerogen_density=1.25, kerogen_K=5., kerogen_G=3., carbon_fraction=.75,
        mixing='hill', dry_model='dual_dem', dem_order='hard_then_soft',
        parameters={k:v[1] for k,v in PARAMETERS.items()}, layers=[],
        salinity_ppm=80000., oil_density=.82, gas_gravity=.8, gor=0.,
        interpolate=False, max_gap_m=1., null_below=-900., clip_negative=False,
        calibration=dict(mode='none', parameters=[], depth_min=0., depth_max=10000.,
            validation_min=0., validation_max=0., max_samples=150, max_evaluations=60,
            weights=dict(vp=1.,vs=1.,rho=1.)))


def finite(value, name, low=None, high=None):
    if isinstance(value, bool):raise ValueError(name+' 必须为数值')
    try:value=float(value)
    except (ValueError, TypeError) as exc:raise ValueError(name+' 必须为数值') from exc
    if not math.isfinite(value) or (low is not None and value<low) or (high is not None and value>high):
        raise ValueError(name+' 超出允许范围')
    return value


def validate_scheme(scheme):
    s=deepcopy(scheme)
    if not str(s.get('name','')).strip():raise ValueError('请填写方案名称')
    lo=finite(s['depth_min'],'起始深度',0);hi=finite(s['depth_max'],'结束深度',0)
    if lo>=hi:raise ValueError('结束深度应大于起始深度')
    if s['mixing'] not in ('voigt','reuss','hill'):raise ValueError('不支持的基质混合方式')
    if s['dry_model'] not in ('cripor','single_dem','dual_dem'):raise ValueError('不支持的骨架模型')
    if s['dem_order'] not in ('hard_then_soft','soft_then_hard'):raise ValueError('不支持的 DEM 顺序')
    if s['mineral_basis'] not in ('weight','volume'):raise ValueError('请选择矿物质量或体积分数')
    for key,(label,_,lower,upper) in PARAMETERS.items():finite(s['parameters'][key],label,lower,upper)
    if not s['minerals']:raise ValueError('至少需要一种矿物')
    if len(s['minerals'])>30:raise ValueError('矿物种类最多 30 种')
    names=[m['name'].strip() for m in s['minerals']]
    if len(set(names))!=len(names) or not all(names):raise ValueError('矿物名称不能为空或重复')
    for mineral in s['minerals']:
        for key in ('K','G','density'):finite(mineral[key],mineral['name']+' '+key,.0001)
        if mineral['unit'] not in ('fraction','%'):raise ValueError('矿物比例单位应为 fraction 或 %')
        if not mineral.get('curve'):finite(mineral['constant'],'矿物比例',0,100 if mineral['unit']=='%' else 1)
    for key,(_,units,_) in FIELDS.items():
        binding=s['bindings'][key]
        if binding['unit'] not in units:raise ValueError(FIELDS[key][0]+' 单位不支持')
        if binding.get('curve') and binding.get('constant') is not None:
            # The source is explicit: a selected curve always takes precedence.
            binding['constant']=None
        if not binding.get('curve') and binding.get('constant') is not None:finite(binding['constant'],FIELDS[key][0])
    finite(s['max_gap_m'],'最大插值间隔',.00001);finite(s['null_below'],'无效值阈值')
    for key,lower,upper in [('salinity_ppm',0,350000),('oil_density',.1,2),('gas_gravity',.1,3),
            ('gor',0,5000),('kerogen_density',.1,5),('kerogen_K',.001,300),('kerogen_G',.001,300),('carbon_fraction',.01,1)]:
        finite(s[key],key,lower,upper)
    previous=lo
    for layer in sorted(s.get('layers',[]),key=lambda x:float(x['from'])):
        a=finite(layer['from'],'分层起点',lo,hi);b=finite(layer['to'],'分层终点',lo,hi)
        if a>=b or a<previous:raise ValueError('分层区间应在建模范围内，且不能重叠')
        previous=b
        for key,value in layer['parameters'].items():
            if key not in PARAMETERS:raise ValueError('未知分层参数：'+key)
            finite(value,PARAMETERS[key][0],PARAMETERS[key][2],PARAMETERS[key][3])
    return s


def convert(values, key, unit):
    values=np.asarray(values,dtype=float).copy()
    if key in ('phi','sw','toc','mineral'):
        if unit=='%':values/=100
    elif key in ('vp','vs'):
        if unit in ('us/ft','us/m'):
            values=np.divide(304800. if unit=='us/ft' else 1e6, values,
                out=np.full_like(values,np.nan),where=values>0)
        elif unit=='km/s':values*=1000
    elif key=='rho' and unit=='kg/m3':values/=1000
    elif key=='pressure':values*=dict(MPa=1.,kPa=.001,psi=.006894757293)[unit]
    return values


def align(depth, values, target, interpolate=False, max_gap=1.):
    """Never bridge explicit missing samples or extrapolate endpoints."""
    order=np.argsort(depth);z=depth[order];v=values[order]
    result=pd.Series(v,index=np.round(z,6)).reindex(np.round(target,6)).to_numpy(dtype=float,copy=True)
    if not interpolate or len(z)<2:return result
    right=np.searchsorted(z,target,side='left');left=right-1
    eligible=(right>0)&(right<len(z))&~np.isfinite(result)
    ii=np.flatnonzero(eligible);l=left[ii];r=right[ii]
    good=np.isfinite(v[l])&np.isfinite(v[r])&((z[r]-z[l])<=max_gap)&(z[r]>z[l])
    ii=ii[good];l=left[ii];r=right[ii]
    result[ii]=v[l]+(v[r]-v[l])*(target[ii]-z[l])/(z[r]-z[l])
    return result


def prepare_data(scheme, well):
    from ..data import read_table,sha256_file
    from ..config import unit_factor
    s=validate_scheme(scheme)
    if s['well']!=well['name']:raise ValueError('方案与井数据不匹配')
    refs={c['name']:c for c in well['curves']};tables={};sources={}
    def read(name):
        if name not in refs:raise ValueError('井中不存在曲线：'+name)
        ref=refs[name];path=Path(ref['path']).resolve();key=str(path)
        if key not in tables:
            tables[key]=read_table(path,dtype={ref['well_column']:'string'} if ref.get('well_column') else None)
            sources[key]=dict(path=key,sha256=sha256_file(path))
        raw=tables[key]
        if ref.get('well_column'):
            raw=raw.loc[raw[ref['well_column']].astype('string').str.strip().eq(ref.get('well_value',well['name'])).fillna(False)]
        z=pd.to_numeric(raw[ref['depth_column']],errors='coerce').to_numpy(dtype=float)*unit_factor('Depth',ref['depth_unit'])
        if not len(z) or not np.isfinite(z).all() or (z<0).any() or pd.Series(z.round(6)).duplicated().any():
            raise ValueError(name+'：深度为空、无效或重复')
        v=pd.to_numeric(raw[ref['column']],errors='coerce').to_numpy(dtype=float,copy=True)
        v[(v<=s['null_below'])|~np.isfinite(v)]=np.nan
        return z,v
    anchor='Depth' if 'Depth' in refs else next((b['curve'] for b in s['bindings'].values() if b.get('curve')),None)
    if anchor is None:raise ValueError('井缺少深度参考曲线')
    z,_=read(anchor);z=np.sort(z[(z>=s['depth_min'])&(z<=s['depth_max'])])
    if not len(z):raise ValueError('建模区间内没有深度样本')
    if len(z)>100000:raise ValueError('单次建模最多 100,000 个样本，请缩小井段')
    data=pd.DataFrame(dict(Depth=z));warnings=[]
    def binding_values(binding,key):
        if binding.get('curve'):
            zz,v=read(binding['curve']);values=align(zz,v,z,s['interpolate'],s['max_gap_m'])
        else:values=np.full(len(z),np.nan if binding.get('constant') is None else float(binding['constant']))
        return convert(values,key,binding['unit'])
    for key in FIELDS:data[key]=binding_values(s['bindings'][key],key)
    fractions=np.column_stack([binding_values(m,'mineral') for m in s['minerals']])
    negative=np.isfinite(fractions)&(fractions<0)
    if s['clip_negative'] and negative.any():
        warnings.append(f'按用户设置将 {int(negative.sum())} 个负矿物比例置零。');fractions[negative]=0.
    invalid=(~np.isfinite(fractions).all(axis=1))|(fractions<0).any(axis=1)|(fractions>1).any(axis=1)
    sums=fractions.sum(axis=1)
    invalid|=(sums<=0)
    if s['normalize_minerals']:
        count=int((~np.isclose(sums,1,atol=.001)&~invalid).sum())
        if count:warnings.append(f'按用户设置归一化 {count} 个样本的无机矿物比例。')
        fractions=np.divide(fractions,sums[:,None],out=np.full_like(fractions,np.nan),where=sums[:,None]>0)
    else:invalid|=~np.isclose(sums,1.,atol=.001)
    # Small roundoff in an accepted unit sum is normalised for exact mixture closure.
    fractions=np.divide(fractions,fractions.sum(axis=1)[:,None],out=np.full_like(fractions,np.nan),where=fractions.sum(axis=1)[:,None]>0)
    densities=np.array([m['density'] for m in s['minerals']],float)
    if s['kerogen']:
        fk=data['toc'].to_numpy()/s['carbon_fraction']
        invalid|=~np.isfinite(fk)|(fk<0)|(fk>=1)
        weights=fractions if s['mineral_basis']=='weight' else fractions*densities
        weights=weights/weights.sum(axis=1)[:,None]
        weights=np.column_stack([weights*(1-fk)[:,None],fk])
        densities=np.append(densities,s['kerogen_density']);volumes=weights/densities
        fractions=volumes/volumes.sum(axis=1)[:,None]
    elif s['mineral_basis']=='weight':
        volumes=fractions/densities;fractions=volumes/volumes.sum(axis=1)[:,None]
    for i in range(fractions.shape[1]):data[f'volume_{i}']=fractions[:,i]
    data['matrix_density']=np.sum(fractions*densities,axis=1)
    reasons=[[] for _ in z]
    def mark(mask,label):
        for i in np.flatnonzero(mask):reasons[i].append(label)
    mark(invalid,'矿物组成/TOC无效')
    for key,low,high in [('phi',0,.99),('sw',0,1),('temperature',0,300),('pressure',0,200)]:
        values=data[key].to_numpy();mark(~np.isfinite(values)|(values<low)|(values>high),FIELDS[key][0]+'无效')
    data['qc_reason']=['；'.join(r) for r in reasons];data['model_valid']=data['qc_reason'].eq('')
    for key in ('vp','vs','rho'):
        data.loc[(data[key]<=0)|~np.isfinite(data[key]),key]=np.nan
    metadata=dict(sources=list(sources.values()),curve_references=deepcopy(well),warnings=warnings,
        alignment='bounded_linear' if s['interpolate'] else 'exact_depth',depth_anchor=anchor)
    return data,metadata


def parameter_arrays(s, depth):
    p={key:np.full(len(depth),float(s['parameters'][key])) for key in PARAMETERS}
    layers=sorted(s.get('layers',[]),key=lambda x:float(x['from']))
    for layer in layers:
        # Half-open intervals; the final modelling endpoint belongs to its layer.
        mask=(depth>=layer['from'])&((depth<layer['to'])|((depth==s['depth_max'])&(depth==layer['to'])))
        for key,value in layer['parameters'].items():p[key][mask]=float(value)
    return p


def forward(data, s):
    from ..vendor.rockphypy.EM import EM
    from ..vendor.rockphypy.BW import BW
    from ..vendor.rockphypy.Fluid import Fluid
    from ..vendor.rockphypy.utils import utils
    result=data.copy();n=len(data);params=parameter_arrays(s,data.Depth.to_numpy())
    kv=[float(m['K']) for m in s['minerals']];gv=[float(m['G']) for m in s['minerals']]
    if s['kerogen']:kv.append(s['kerogen_K']);gv.append(s['kerogen_G'])
    volumes=data[[f'volume_{i}' for i in range(len(kv))]].to_numpy()
    index=dict(voigt=0,reuss=1,hill=2)[s['mixing']]
    with np.errstate(all='ignore'):
        km=EM.VRH(volumes,np.array(kv))[index];gm=EM.VRH(volumes,np.array(gv))[index]
        rb,kb=BW.rho_K_brine(data.temperature.to_numpy(),data.pressure.to_numpy(),s['salinity_ppm']/1e6)
        if s['gor']>0:ro,ko=BW.rho_K_go(data.pressure.to_numpy(),data.temperature.to_numpy(),s['oil_density'],s['gas_gravity'],s['gor'])
        else:ro,ko=BW.rho_K_oil(data.pressure.to_numpy(),data.temperature.to_numpy(),s['oil_density'])
        sw=data.sw.to_numpy();rf=sw*rb+(1-sw)*ro;kf=1/(sw/kb+(1-sw)/ko)
    phi=data.phi.to_numpy();valid=data.model_valid.to_numpy().copy()
    valid&=np.isfinite(km)&np.isfinite(gm)&(km>0)&(gm>0)&np.isfinite(kf)&(kf>0)&np.isfinite(rf)&(rf>0)
    kd=np.full(n,np.nan);gd=kd.copy()
    def dem(k,g,alpha,p):
        if p==0:return k,g
        kk,gg,pp=EM.Berryman_DEM(float(k),float(g),0.,0.,float(alpha),float(p))
        return float(kk[-1]),float(gg[-1])
    if s['dry_model']=='cripor':
        valid&=phi<params['critical_porosity']
        kd[valid],gd[valid]=EM.cripor(km[valid],gm[valid],phi[valid],params['critical_porosity'][valid])
    else:
        for i in np.flatnonzero(valid):
            if s['dry_model']=='single_dem':kd[i],gd[i]=dem(km[i],gm[i],params['alpha_hard'][i],phi[i])
            else:
                soft=phi[i]*params['soft_fraction'][i];hard=phi[i]-soft
                first,second,a,b=(hard/(1-soft),soft,params['alpha_hard'][i],params['alpha_soft'][i]) if s['dem_order']=='hard_then_soft' else (soft/(1-hard),hard,params['alpha_soft'][i],params['alpha_hard'][i])
                k,g=dem(km[i],gm[i],a,first);kd[i],gd[i]=dem(k,g,b,second)
    ks=np.full(n,np.nan);gs=ks.copy()
    nonzero=valid&(phi>0)
    with np.errstate(all='ignore'):
        ks[nonzero],gs[nonzero]=Fluid.Gassmann(kd[nonzero],gd[nonzero],km[nonzero],kf[nonzero],phi[nonzero])
        ks[valid&(phi==0)]=km[valid&(phi==0)];gs[valid&(phi==0)]=gm[valid&(phi==0)]
        rho=(1-phi)*data.matrix_density.to_numpy()+phi*rf
        vp,vs=utils.V(ks,gs,rho)
        valid&=np.isfinite(vp)&np.isfinite(vs)&(vp>0)&(vs>0)&np.isfinite(rho)&(rho>0)&(ks>0)&(gs>0)
        outputs=dict(VP_RP=vp,VS_RP=vs,DEN_RP=rho,AC_RP=304800/vp,IP_RP=vp*rho,
            VPVS_RP=vp/vs,K_RP=ks,G_RP=gs,K_matrix=km,G_matrix=gm,K_dry=kd,G_dry=gd,K_fluid=kf,rho_fluid=rf)
    for key,values in outputs.items():result[key]=np.where(valid,values,np.nan)
    failed=data.model_valid.to_numpy()&~valid
    result.loc[failed,'qc_reason']=result.loc[failed,'qc_reason'].map(lambda x:(x+'；' if x else '')+'骨架或流体计算无效')
    result['model_valid']=valid
    for key,model in [('vp','VP_RP'),('vs','VS_RP'),('rho','DEN_RP')]:result[key+'_error']=result[model]-result[key]
    return result


def metrics(frame,mask=None):
    result={};mask=np.ones(len(frame),bool) if mask is None else np.asarray(mask,bool)
    for key,model in [('vp','VP_RP'),('vs','VS_RP'),('rho','DEN_RP')]:
        obs=frame[key].to_numpy();pred=frame[model].to_numpy();valid=mask&np.isfinite(obs)&np.isfinite(pred)
        count=int(valid.sum())
        if count:
            err=pred[valid]-obs[valid];den=np.sum((obs[valid]-obs[valid].mean())**2)
            result[key]=dict(count=count,rmse=float(np.sqrt(np.mean(err**2))),bias=float(err.mean()),
                r2=float(1-np.sum(err**2)/den) if count>1 and den>0 else None)
    return result


def calibrate(data,s,progress=None):
    """Fit only selected global parameters; layers remain explicit fixed overrides."""
    c=s['calibration'];mode=c.get('mode','none')
    if mode=='none':return s,dict(mode='none',evaluations=0,scan=[])
    if mode not in ('scan','optimize'):raise ValueError('未知标定方式')
    bounds=c.get('parameters',[])
    if not 1<=len(bounds)<=3 or (mode=='scan' and len(bounds)>2):raise ValueError('扫描选择 1～2 个参数，自动标定选择 1～3 个参数')
    if len({b['name'] for b in bounds})!=len(bounds):raise ValueError('标定参数重复')
    active={'critical_porosity'} if s['dry_model']=='cripor' else {'alpha_hard'} if s['dry_model']=='single_dem' else {'alpha_hard','alpha_soft','soft_fraction'}
    for b in bounds:
        key=b['name']
        if key not in active:raise ValueError('标定参数不适用于当前骨架模型：'+key)
        low=finite(b['min'],key+' 下限',PARAMETERS[key][2],PARAMETERS[key][3]);high=finite(b['max'],key+' 上限',PARAMETERS[key][2],PARAMETERS[key][3])
        if low>=high:raise ValueError('标定参数上限应大于下限')
        if any(key in layer['parameters'] for layer in s.get('layers',[])):raise ValueError('标定参数已被分层覆盖，请先移除对应分层覆盖：'+key)
    lo=finite(c['depth_min'],'标定起点',s['depth_min'],s['depth_max']);hi=finite(c['depth_max'],'标定终点',lo,s['depth_max'])
    vlo=finite(c.get('validation_min',0),'验证起点',0);vhi=finite(c.get('validation_max',0),'验证终点',0)
    if hi<=lo:raise ValueError('标定井段不能为空')
    if vhi<vlo:raise ValueError('验证终点不能小于验证起点')
    if vhi>vlo and not (vhi<lo or vlo>hi):raise ValueError('验证井段不能与标定井段重叠')
    if vhi>vlo and (vlo<s['depth_min'] or vhi>s['depth_max']):raise ValueError('验证井段应在建模范围内')
    mask=data.model_valid.to_numpy()&(data.Depth>=lo)&(data.Depth<=hi)
    selected=np.flatnonzero(mask)
    maximum=int(finite(c.get('max_samples',150),'标定最大样本',3,2000))
    if len(selected)>maximum:selected=selected[np.unique(np.linspace(0,len(selected)-1,maximum).round().astype(int))]
    subset=data.iloc[selected].reset_index(drop=True)
    weights={k:finite(c.get('weights',{}).get(k,1),k+' 标定权重',0,100) for k in ('vp','vs','rho')}
    observations={k:np.isfinite(subset[k].to_numpy())&(subset[k].to_numpy()>0) for k in weights if weights[k]>0}
    observations={k:v for k,v in observations.items() if v.any()}
    if not observations:raise ValueError('标定井段没有有效实测参考曲线，或所有权重为零')
    if sum(int(v.sum()) for v in observations.values())<=len(bounds):raise ValueError('有效参考样本不足以标定所选参数')
    best=[np.inf,None];history=[];evaluations=[0]
    def residual(x):
        candidate=deepcopy(s)
        for b,value in zip(bounds,x):candidate['parameters'][b['name']]=float(value)
        modeled=forward(subset,candidate);parts=[]
        for key,valid in observations.items():
            pred=modeled[dict(vp='VP_RP',vs='VS_RP',rho='DEN_RP')[key]].to_numpy()[valid];obs=subset[key].to_numpy()[valid]
            # Each curve's weight is independent of its number of valid samples.
            err=(pred-obs)/np.maximum(np.abs(obs),1e-12)
            parts.append(np.where(np.isfinite(err),err,1e3)*np.sqrt(weights[key]/len(obs)))
        r=np.concatenate(parts);score=float(np.dot(r,r));evaluations[0]+=1
        history.append(dict(parameters={b['name']:float(v) for b,v in zip(bounds,x)},score=score))
        if score<best[0]:best[:]=[score,candidate]
        if progress and (evaluations[0]==1 or evaluations[0]%5==0):progress(f'参数标定：已计算 {evaluations[0]} 组')
        return r
    convergence={}
    if mode=='scan':
        from itertools import product
        grids=[np.linspace(b['min'],b['max'],int(finite(b.get('steps',7),'扫描步数',2,30))) for b in bounds]
        if math.prod(len(g) for g in grids)>400:raise ValueError('扫描组合最多 400 组')
        for x in product(*grids):residual(x)
    else:
        from scipy.optimize import least_squares
        lower=[b['min'] for b in bounds];upper=[b['max'] for b in bounds]
        x=np.clip([s['parameters'][b['name']] for b in bounds],lower,upper)
        fit=least_squares(residual,x,bounds=(lower,upper),diff_step=.001,
            max_nfev=int(finite(c.get('max_evaluations',60),'最大优化迭代',5,200)),xtol=1e-5,ftol=1e-5,gtol=1e-6)
        convergence=dict(converged=bool(fit.success),message=str(fit.message))
    if best[1] is None or best[0]>=1e5:raise ValueError('标定范围内没有可用模型，请检查参数及输入数据')
    scores=np.array([h['score'] for h in history])
    if len(scores)>1 and np.ptp(scores)<=1e-12*max(1.,abs(float(scores.min()))):
        raise ValueError('参考曲线对所选参数缺乏敏感性，无法标定；请调整参考曲线、参数或标定井段')
    return best[1],dict(mode=mode,evaluations=evaluations[0],sample_count=len(subset),
        sample_depths=subset.Depth.tolist(),reference_counts={k:int(v.sum()) for k,v in observations.items()},
        weights=weights,score=best[0],scan=history,**convergence)


def preview_sample(data,limit):
    """Bound numerical work while recording gaps hidden by depth decimation."""
    from ..well_points import gap_limit
    limit=int(finite(limit,'预览最大样点数',8,2000))
    selected=np.unique(np.linspace(0,len(data)-1,min(limit,len(data))).round().astype(int))
    if not data.iloc[selected].model_valid.any() and data.model_valid.any():
        selected[len(selected)//2]=int(np.flatnonzero(data.model_valid.to_numpy())[0]);selected=np.unique(np.sort(selected))
    depth=data.Depth.to_numpy();large_gap=np.diff(depth)>gap_limit(depth)
    breaks={}
    for key in ('model','vp','vs','rho'):
        valid=data.model_valid.to_numpy() if key=='model' else np.isfinite(data[key].to_numpy())
        indices=[]
        for position,(left,right) in enumerate(zip(selected[:-1],selected[1:]),1):
            if (~valid[left:right+1]).any() or large_gap[left:right].any():indices.append(position)
        breaks[key]=indices
    return data.iloc[selected].reset_index(drop=True),dict(total_rows=len(data),
        input_valid_rows=int(data.model_valid.sum()),sampled_rows=len(selected),limit=limit,breaks=breaks)


def run_workflow(scheme,well,progress=None,preview_limit=None):
    from .. import __version__
    s=validate_scheme(scheme)
    if progress:progress('读取井曲线并检查单位、深度和矿物组成…')
    data,provenance=prepare_data(s,well)
    if not data.model_valid.any():raise ValueError('建模区间没有有效样本：'+ '；'.join(data.qc_reason.unique()[:5]))
    preview=None
    if preview_limit is not None:
        # Interactive preview is always forward-only; calibration remains explicit.
        s['calibration']['mode']='none'
        data,preview=preview_sample(data,preview_limit)
    if progress:progress('构建矿物—骨架—流体基线模型…')
    initial=forward(data,s);adopted,calibration=calibrate(data,s,progress)
    result=forward(data,adopted) if calibration['mode']!='none' else initial
    if not result.model_valid.any():raise ValueError('模型未产生有效结果，请检查骨架和流体参数')
    c=s['calibration'];z=result.Depth.to_numpy();partition=np.full(len(z),'未参与标定',object)
    if calibration['mode']!='none':
        partition[(z>=c['depth_min'])&(z<=c['depth_max'])]='标定区间'
        partition[np.isin(z,calibration['sample_depths'])]='标定样本'
        if c['validation_max']>c['validation_min']:
            partition[(z>=c['validation_min'])&(z<=c['validation_max'])]='独立验证'
    result['partition']=partition
    reports=dict(all=metrics(result),before=metrics(initial),
        calibration=metrics(result,np.isin(z,calibration.get('sample_depths',[]))),validation=metrics(result,partition=='独立验证'))
    warnings=provenance.pop('warnings')
    warnings.append('Gassmann 为低频各向同性基线近似；与声波测井的频率差异、页岩各向异性需结合资料解释。')
    if s['mineral_basis']=='weight':warnings.append('矿物质量分数已按组分密度换算为体积分数。')
    if s['kerogen']:warnings.append('TOC 按干固体总质量中的碳质量分数解释，干酪根质量分数 = TOC / 碳比例。')
    invalid=int((~result.model_valid).sum())
    if invalid:warnings.append(f'{invalid} 个无效样本保留原深度和质控原因，成果值留空。')
    if calibration.get('converged') is False:warnings.append('自动标定达到迭代限制，保留已评估的最优参数，请检查收敛信息。')
    # JSON null represents a gap; never emit nonstandard NaN/Infinity tokens.
    clean=result.replace([np.inf,-np.inf],np.nan).astype(object)
    clean=clean.where(pd.notna(clean),None)
    parameters=dict(scheme=s,adopted_scheme=adopted,provenance=provenance)
    source_hash=hashlib.sha256(json.dumps(parameters,sort_keys=True,ensure_ascii=False,allow_nan=False).encode('utf-8')).hexdigest()
    vendor=Path(__file__).resolve().parents[1]/'vendor/rockphypy'
    return dict(schema_version=1,status='completed',kind='well_log',model_id='well_log.baseline',
        title=well['name']+' · '+s['name'],well=well['name'],scheme_name=s['name'],software_version=__version__,
        parameters=parameters,provenance_sha256=source_hash,
        library=dict(name='RockPhyPy',source=json.loads((vendor/'SOURCE.json').read_text(encoding='utf-8')),
            kernel_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (vendor/'EM.py',vendor/'Fluid.py',vendor/'BW.py',vendor/'utils.py')}),
        outputs=[],table=dict(columns=list(clean.columns),rows=clean.values.tolist()),units=OUTPUT_UNITS,
        summary=dict(rows=len(result),modeled_rows=int(result.model_valid.sum()),invalid_rows=invalid),
        metrics=reports,calibration=calibration,warnings=warnings,preview=preview)
