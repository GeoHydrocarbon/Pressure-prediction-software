"""Well geometry and bounded, on-demand curves for the SEG-Y browser."""
from pathlib import Path
import hashlib
import json
import numpy as np
from .well_spatial import read_asset,interpolate_path
from .config import Project,unit_factor
from .time_depth import TimeDepth
from .data import read_table


def serial(values):
    values=np.asarray(values,dtype=float)
    return np.where(np.isfinite(values),values,None).tolist()


def curve_id(well,ref):
    return hashlib.sha256(json.dumps([well,*[ref.get(k,'') for k in ('path','column','name','well_column','well_value')]],ensure_ascii=False).encode()).hexdigest()[:24]


def project_points(points,line,axis):
    """Closest bounded polyline projection, interpolating the section abscissa."""
    points=np.asarray(points,float).reshape(-1,2);line=np.asarray(line,float);axis=np.asarray(axis,float)
    distance=np.full(len(points),np.inf);abscissa=np.full(len(points),np.nan)
    for a,b,xa,xb in zip(line[:-1],line[1:],axis[:-1],axis[1:]):
        vector=b-a;norm=np.dot(vector,vector)
        if norm<1e-20:continue
        t=((points-a)@vector)/norm
        valid=np.isfinite(points).all(axis=1)&(t>=-1e-10)&(t<=1+1e-10)
        d=np.linalg.norm(points-(a+np.clip(t,0,1)[:,None]*vector),axis=1)
        selected=np.isfinite(d)&((d<distance-1e-10)|(np.isclose(d,distance,atol=1e-10,rtol=0)&valid&~np.isfinite(abscissa)))
        distance[selected]=d[selected];abscissa[selected]=np.where(valid[selected],xa+np.clip(t[selected],0,1)*(xb-xa),np.nan)
    return abscissa,distance


class WellContext:
    def __init__(self,dataset):
        self.dataset=dataset;self.warnings=[];self.wells={};self.tables={};self.refs={}
        heads={};paths={}
        for asset in dataset.config.get('well_assets',[]):
            if asset['kind'] not in ('coordinates','trajectory'):continue
            try:
                _,frame=read_asset(asset)
                target=heads if asset['kind']=='coordinates' else paths
                for name,group in frame.groupby('well',sort=False):
                    group=group.copy()
                    if asset['kind']=='trajectory':group['relative']=asset.get('trajectory_mode')=='survey' or asset.get('relative',False)
                    target.setdefault(str(name),[]).append(group)
            except Exception as exc:self.warnings.append('井空间数据读取失败：'+str(exc))
        libraries={w['name']:w for w in dataset.config.get('well_library',[])}
        for name in dict.fromkeys([*heads,*paths,*libraries]):
            head=None;trajectory=None
            if name in heads:
                xy=np.concatenate([g[['x','y']].to_numpy(float) for g in heads[name]])
                if np.isfinite(xy).all() and np.allclose(xy,xy[0],atol=1e-6,rtol=0):head=xy[0]
                else:self.warnings.append(f'{name} 的井口坐标冲突或无效，未采用')
            if name in paths:
                import pandas as pd
                trajectory=pd.concat(paths[name],ignore_index=True).sort_values('depth',kind='stable')
                if trajectory.relative.any():
                    if head is None:trajectory=None;self.warnings.append(f'{name} 的相对轨迹缺少唯一井口坐标')
                    else:trajectory.loc[trajectory.relative,['x','y']]+=head
                if trajectory is not None:
                    if trajectory.depth.duplicated().any():
                        groups=trajectory.groupby('depth',sort=True)
                        for _,group in groups:
                            xy=group[['x','y']].to_numpy(float)
                            if not np.isfinite(xy).all() or not np.allclose(xy,xy[0],atol=1e-6,rtol=0):trajectory.loc[group.index,['x','y']]=np.nan
                        trajectory=trajectory.drop_duplicates('depth',keep='first')
                    if head is None and len(trajectory):
                        first=trajectory.iloc[0][['x','y']].to_numpy(float)
                        if np.isfinite(first).all():head=first
            if head is None:continue
            library=libraries.get(name,{})
            well=dict(name=name,head=head,trajectory=trajectory,time_depth_id=library.get('time_depth_id'),curves=[])
            for ref in library.get('curves',[]):
                if ref['name']=='Depth':continue
                ident=curve_id(name,ref);self.refs[ident]=(name,ref)
                well['curves'].append(dict(id=ident,name=ref['name'],unit=ref.get('unit',''),mode='points' if ref.get('role')=='measurement' or ref['name']=='Pp' and ref.get('role')!='derived' else 'line'))
            self.wells[name]=well

    def map_coordinates(self,xy,mode):
        xy=np.asarray(xy,float)
        if mode=='xy':return xy
        ds=self.dataset
        if not ds.xy_available:return np.full(xy.shape,np.nan)
        # An affine survey grid preserves smooth paths without snapping to traces.
        if not hasattr(self,'grid_transform'):
            rows=ds.map_indices
            source=np.column_stack([ds.x[rows],ds.y[rows]]);center=source.mean(axis=0);scale=np.maximum(np.ptp(source,axis=0),1.)
            design=np.column_stack([(source-center)/scale,np.ones(len(rows))])
            target=np.column_stack([ds.crossline[rows],ds.inline[rows]])
            coefficients,_,rank,_=np.linalg.lstsq(design,target,rcond=None)
            self.grid_transform=(center,scale,coefficients,rank==3 and np.max(np.abs(design@coefficients-target))<.01)
        center,scale,coefficients,affine=self.grid_transform
        if affine:
            return np.column_stack([(xy-center)/scale,np.ones(len(xy))])@coefficients
        distance,nearest=ds.tree('xy')[0].query(xy)
        mapped=np.column_stack([ds.crossline[nearest],ds.inline[nearest]]).astype(float)
        mapped[distance>ds.config.get('well_max_distance',25.)]=np.nan
        return mapped

    def map_wells(self):
        result=[]
        for well in self.wells.values():
            head=well['head'];item=dict(name=well['name'],x=float(head[0]),y=float(head[1]))
            grid=self.map_coordinates(head[None,:],'lines')[0]
            if np.isfinite(grid).all():item.update(crossline=float(grid[0]),inline=float(grid[1]))
            path=well['trajectory']
            if path is not None and len(path):
                # Preserve gaps; maps are bounded independently of survey size.
                rows=np.linspace(0,len(path)-1,min(5000,len(path)),dtype=int)
                xy=path[['x','y']].to_numpy(float)[rows];good=np.isfinite(xy).all(axis=1)
                grid=np.full_like(xy,np.nan)
                if good.any():grid[good]=self.map_coordinates(xy[good],'lines')
                item.update(trajectory_xy=serial(xy),trajectory_lines=serial(grid))
            result.append(item)
        return result,self.warnings

    def time_depth(self,well):
        ident=well.get('time_depth_id')
        entry=next((r for r in self.dataset.config.get('time_depth_relations',[]) if r['id']==ident),None)
        if entry is None:return None
        return TimeDepth(Project(Path(self.dataset.config.get('project_path',self.dataset.source)),{'time_depth':entry['config']}))

    def times(self,td,depth):
        times=td.inverse(depth)*1000.
        original=td.config['time_convention'];target=self.dataset.config.get('time_convention','twt')
        if original!=target:times*=2. if target=='twt' else .5
        return times

    def curve_values(self,ident):
        name,ref=self.refs[ident];key=(ref['path'],ref.get('well_column',''))
        if key not in self.tables:self.tables[key]=read_table(Path(ref['path']),dtype={ref['well_column']:'string'} if ref.get('well_column') else None)
        raw=self.tables[key]
        if ref.get('well_column'):raw=raw.loc[raw[ref['well_column']].astype('string').str.strip().eq(ref.get('well_value',name)).fillna(False)]
        import pandas as pd
        depth=pd.to_numeric(raw[ref['depth_column']],errors='coerce').to_numpy(float)*unit_factor('Depth',ref.get('depth_unit','m'))
        values=pd.to_numeric(raw[ref['column']],errors='coerce').to_numpy(float)
        order=np.argsort(depth,kind='stable');depth=depth[order];values=values[order]
        if len(depth)>20000:
            # Keep real observations; no value interpolation or fabricated samples.
            selected=np.linspace(0,len(depth)-1,20000,dtype=int)
            diff=np.diff(depth);positive=diff[np.isfinite(diff)&(diff>0)]
            gap=5*np.median(positive) if len(positive) else np.inf
            bad=~np.isfinite(depth)|~np.isfinite(values)|np.r_[False,(~np.isfinite(diff))|(diff<=0)|(diff>gap)]
            prefix=np.r_[0,np.cumsum(bad)]
            breaks=np.flatnonzero(prefix[selected[1:]+1]-prefix[selected[:-1]+1]>0)+1
            depth=depth[selected];values=values[selected]
            depth=np.insert(depth,breaks,np.nan);values=np.insert(values,breaks,np.nan)
        return depth,values

    def section(self,line,axis,coordinate_mode,selected_curves):
        result=[];warnings=[];selected=set(selected_curves or [])
        for well in self.wells.values():
            head=self.map_coordinates(well['head'][None,:],coordinate_mode)
            hx,hd=project_points(head,line,axis)
            item=dict(name=well['name'],head_x=serial(hx)[0],head_distance=serial(hd)[0],curves=well['curves'],curve_data=[],track=None)
            try:td=self.time_depth(well)
            except Exception as exc:td=None;warnings.append(well['name']+' 的时深关系不可用：'+str(exc))
            trajectory=well['trajectory']
            if trajectory is not None and len(trajectory)>=2:
                depth=trajectory.depth.to_numpy(float)
                # Densify only the geometry, keeping invalid survey intervals broken.
                dense=np.unique(np.r_[depth,np.linspace(depth[0],depth[-1],min(5000,max(2,int((depth[-1]-depth[0])/2)+1)))])
                # Include exact spatial intersections even when the display corridor is zero.
                stations=trajectory[['x','y']].to_numpy(float);valid_station=np.isfinite(stations).all(axis=1)
                mapped_station=np.full_like(stations,np.nan)
                if valid_station.any():mapped_station[valid_station]=self.map_coordinates(stations[valid_station],coordinate_mode)
                start=mapped_station[:-1];vector=np.diff(mapped_station,axis=0);crossings=[]
                for a,b in zip(np.asarray(line)[:-1],np.asarray(line)[1:]):
                    edge=b-a;denominator=vector[:,0]*edge[1]-vector[:,1]*edge[0]
                    safe=np.where(np.abs(denominator)>1e-12,denominator,np.nan);offset=a-start
                    t=(offset[:,0]*edge[1]-offset[:,1]*edge[0])/safe
                    u=(offset[:,0]*vector[:,1]-offset[:,1]*vector[:,0])/safe
                    valid=np.isfinite(t)&np.isfinite(u)&(t>=0)&(t<=1)&(u>=0)&(u<=1)
                    if valid.any():crossings.extend((depth[:-1][valid]+t[valid]*np.diff(depth)[valid]).tolist())
                if crossings:dense=np.unique(np.r_[dense,crossings])
                xy=interpolate_path(trajectory,dense);good=np.isfinite(xy).all(axis=1);mapped=np.full_like(xy,np.nan)
                if good.any():mapped[good]=self.map_coordinates(xy[good],coordinate_mode)
                x,distance=project_points(mapped,line,axis)
                finite=distance[np.isfinite(distance)&np.isfinite(x)]
                item['geometry_distance']=float(finite.min()) if len(finite) else None
                if td is not None:
                    try:
                        time=self.times(td,dense)
                        item['track']=dict(x=serial(x),distance=serial(distance),time=serial(time))
                    except Exception as exc:
                        warnings.append(well['name']+' 的轨迹时间转换失败：'+str(exc));td=None
            for ref in well['curves']:
                if ref['id'] not in selected:continue
                if td is None:
                    warnings.append(well['name']+' / '+ref['name']+' 缺少已应用的时深关系，未叠加');continue
                try:
                    depth,values=self.curve_values(ref['id']);time=self.times(td,depth)
                    if trajectory is not None:
                        xy=interpolate_path(trajectory,depth);good=np.isfinite(xy).all(axis=1);mapped=np.full_like(xy,np.nan)
                        if good.any():mapped[good]=self.map_coordinates(xy[good],coordinate_mode)
                        x,distance=project_points(mapped,line,axis)
                    else:x=np.full(len(depth),hx[0]);distance=np.full(len(depth),hd[0])
                    # Do not draw lines across missing records or large logging gaps.
                    diff=np.diff(depth);positive=diff[np.isfinite(diff)&(diff>0)]
                    breaks=np.flatnonzero((diff<=0)|(diff>5*np.median(positive)))+1 if len(positive) else np.array([],int)
                    if len(breaks):
                        time=np.insert(time,breaks,np.nan);values=np.insert(values,breaks,np.nan);x=np.insert(x,breaks,np.nan);distance=np.insert(distance,breaks,np.nan)
                    item['curve_data'].append(dict(id=ref['id'],x=serial(x),distance=serial(distance),time=serial(time),values=serial(values)))
                except Exception as exc:warnings.append(well['name']+' / '+ref['name']+'：'+str(exc))
            result.append(item)
        return result,warnings
