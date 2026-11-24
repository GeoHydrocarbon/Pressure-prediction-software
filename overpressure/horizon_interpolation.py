"""Bounded spatial interpolation of horizon times on existing seismic traces."""
from pathlib import Path
from uuid import uuid4
import json
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay,QhullError
from .horizons import HorizonMap,geometry_hash
from .data import sha256_file
from .jobs import save_json


def interpolate_times(index,times,*,max_span,coordinates='lines',point_types=None):
    if coordinates not in ('lines','xy'):raise ValueError('插值坐标方式必须为 lines 或 xy')
    if isinstance(max_span,bool) or not np.isfinite(max_span) or not 0<float(max_span)<=1e9:raise ValueError('最大插值跨度必须为有限正数，且不超过 1e9')
    source=np.asarray(times,dtype=float)
    if source.shape!=index.inline.shape:raise ValueError('层位与地震道数不一致')
    kinds=np.where(np.isfinite(source),1,0).astype(np.uint8) if point_types is None else np.asarray(point_types).copy()
    if kinds.shape!=source.shape or not np.isin(kinds,[0,1,2]).all() or not np.array_equal(kinds>0,np.isfinite(source)):raise ValueError('层位原始点/插值点标记无效')
    kinds=kinds.astype(np.uint8)
    if coordinates=='xy':
        if index.x is None or index.y is None:raise ValueError('当前地震索引没有 XY 坐标，请重新载入成果')
        if index.coordinate_units is not None and not np.isin(index.coordinate_units,[0,1]).all():raise ValueError('XY 插值仅支持一致的平面长度坐标，不支持经纬度；可改用线号网格')
        xy=np.column_stack((index.x,index.y)).astype(float)
    else:xy=np.column_stack((index.crossline,index.inline)).astype(float)
    finite_xy=np.isfinite(xy).all(axis=1);known=(kinds==1)&finite_xy
    if known.sum()<3:raise ValueError('至少需要 3 个覆盖二维区域的原始层位点；共线点不能进行空间线性插值')
    if known.sum()>500000:raise ValueError('本版单层插值最多使用 50 万个原始匹配点，请先缩小层位范围')
    points,first,inverse=np.unique(xy[known],axis=0,return_index=True,return_inverse=True)
    seed_values=source[known];lo=np.full(len(points),np.inf);hi=np.full(len(points),-np.inf)
    np.minimum.at(lo,inverse,seed_values);np.maximum.at(hi,inverse,seed_values)
    if ((hi-lo)>1e-6).any():raise ValueError('相同平面坐标存在冲突的层位时间，请核对坐标或改用线号网格')
    values=seed_values[first]
    try:tri=Delaunay(points)
    except QhullError as exc:raise ValueError('原始层位点无法形成有效二维三角网；请核对坐标，避免共线或重合点') from exc
    vertices=points[tri.simplices]
    longest=np.maximum.reduce([np.linalg.norm(vertices[:,a]-vertices[:,b],axis=1) for a,b in ((0,1),(1,2),(2,0))])
    output=source.copy();candidates=np.flatnonzero(~np.isfinite(source)&finite_xy)
    counts=dict(original_points=int((kinds==1).sum()),previous_interpolated_points=int((kinds==2).sum()),new_interpolated_points=0,outside_convex_hull=0,span_rejected=0,numeric_rejected=0,invalid_coordinate_points=int((~np.isfinite(source)&~finite_xy).sum()))
    for start in range(0,len(candidates),50000):
        rows=candidates[start:start+50000];query=xy[rows];simplex=tri.find_simplex(query);inside=simplex>=0
        counts['outside_convex_hull']+=int((~inside).sum());rows=rows[inside];query=query[inside];simplex=simplex[inside]
        allowed=longest[simplex]<=float(max_span)*(1+1e-12)
        counts['span_rejected']+=int((~allowed).sum());rows=rows[allowed];query=query[allowed];simplex=simplex[allowed]
        transform=tri.transform[simplex];weights=np.einsum('ijk,ik->ij',transform[:,:2,:],query-transform[:,2,:])
        weights=np.column_stack((weights,1-weights.sum(axis=1)))
        result=(values[tri.simplices[simplex]]*weights).sum(axis=1);valid=np.isfinite(result)&np.isfinite(weights).all(axis=1)&(weights>=-1e-9).all(axis=1)
        counts['numeric_rejected']+=int((~valid).sum());output[rows[valid]]=result[valid];kinds[rows[valid]]=2;counts['new_interpolated_points']+=int(valid.sum())
    counts.update(interpolated_points=int((kinds==2).sum()),remaining_unmapped_points=int((~np.isfinite(output)).sum()),mapped_trace_values=int(np.isfinite(output).sum()))
    return output,kinds,counts


def interpolate_horizon(index,mapped_path,layer_name,*,new_name,max_span,coordinates='lines'):
    from .volume_reader import require_fresh
    require_fresh(index.path);source_path=Path(mapped_path).resolve();horizon=HorizonMap.load(source_path,index)
    if not isinstance(new_name,str) or not new_name.strip() or len(new_name.strip())>160 or new_name.strip()==layer_name:raise ValueError('请输入与原层位不同的新名称（不超过 160 字）')
    new_name=new_name.strip();position=horizon.names.index(layer_name)
    times,kinds,counts=interpolate_times(index,horizon.layer(layer_name),point_types=horizon.point_types[position],max_span=max_span,coordinates=coordinates)
    require_fresh(index.path)
    folder=index.path.parent/'horizons';folder.mkdir(exist_ok=True);target=folder/(uuid4().hex+'_interpolated.npz');csv=target.with_suffix('.csv')
    temporary=target.with_name(target.stem+'.partial.npz')
    np.savez_compressed(temporary,names=np.array([new_name]),times_ms=times[None,:],point_types=kinds[None,:],geometry=np.array(geometry_hash(index)));temporary.replace(target)
    chosen=np.isfinite(times);table=pd.DataFrame(dict(inline=index.inline[chosen],crossline=index.crossline[chosen],time_ms=times[chosen],horizon=new_name,point_type=np.where(kinds[chosen]==1,'original','interpolated')))
    if index.x is not None:table['x']=index.x[chosen];table['y']=index.y[chosen]
    temp_csv=csv.with_suffix('.csv.tmp');table.to_csv(temp_csv,index=False,encoding='utf-8-sig');temp_csv.replace(csv)
    config=dict(path=str(csv),display_name=new_name,coordinates='lines',columns=dict(inline='inline',crossline='crossline',time='time_ms',name='horizon'),time_unit='ms',time_convention=index.time_convention,max_distance=0.,point_type_column='point_type',interpolation_report=str(target.with_suffix('.json')))
    local=(times-index.t0_ms)/index.interval_ms-index.offsets
    report=dict(path=str(target),names=[new_name],source=str(csv),source_sha256=sha256_file(csv),settings=config,records=len(table),status_counts={'matched':len(table)},outside_time_values=int((chosen&((local< -1e-7)|(local>index.ns-1+1e-7))).sum()),spatial_interpolation=True,derived=True,interpolation_report=str(target.with_suffix('.json')),original_map=str(source_path),original_map_sha256=sha256_file(source_path),original_layer=layer_name,method='linear_delaunay',coordinates=coordinates,max_span=float(max_span),range_rule='inside convex hull of original mapped points; all triangle edges <= max_span; existing traces only; no pressure interpolation',fault_constraints=False,**counts)
    save_json(target.with_suffix('.json'),report);return report
