from __future__ import annotations
import numpy as np
from .config import Project,unit_factor
from .data import read_table


class TimeDepth:
    """Convert valid input samples; legacy manual time limits are ignored."""
    def __init__(self,project:Project,*,time_bounds_seconds=None):
        self.config=dict(project.settings['time_depth']);cfg=self.config
        self.seconds_per_unit={'s':1.,'ms':.001}.get(cfg['time_unit'])
        if self.seconds_per_unit is None:raise ValueError('时深关系时间单位必须为 s 或 ms')
        self.depth_factor=unit_factor('Depth',cfg['depth_unit'])
        if cfg['time_convention'] not in ('twt','owt'):raise ValueError('时深关系时间类型必须为 twt 或 owt')
        self.bounds=None;self.observed_bounds=None
        if cfg['method']=='polynomial':
            coefficients=np.asarray(cfg['coefficients'],dtype=float)
            if coefficients.ndim!=1 or not 2<=len(coefficients)<=4 or not np.isfinite(coefficients).all():raise ValueError('多项式仅支持 1 至 3 次，系数必须为有限数')
            self.polynomial=np.polynomial.Polynomial(coefficients)
        elif cfg['method']=='table':
            table=read_table(project.resolve(cfg['table_file']))
            if not {'time','depth'}<=set(table.columns):raise ValueError('时深表必须包含 time 和 depth 列')
            pairs=table[['time','depth']].to_numpy(dtype=float)
            if len(pairs)<2 or not np.isfinite(pairs).all() or not (np.diff(pairs,axis=0)>0).all():raise ValueError('时深表至少两行，时间和深度必须按原始行顺序严格递增且有限')
            self.times,self.depths=pairs.T;self.bounds=self.times[[0,-1]].copy()
        else:raise ValueError('时深方式必须为 polynomial 或 table')
        if time_bounds_seconds is not None:
            bounds=np.asarray(time_bounds_seconds,dtype=float)/self.seconds_per_unit
            if bounds.shape!=(2,) or not np.isfinite(bounds).all() or bounds[1]<bounds[0]:raise ValueError('输入地震时间范围必须为非递减的有限数')
            self.bounds=bounds if self.bounds is None else np.array([max(bounds[0],self.bounds[0]),min(bounds[1],self.bounds[1])])

    def check_polynomial(self,bounds):
        derivative=self.polynomial.deriv();points=list(bounds)
        for root in derivative.deriv().roots():
            if np.isreal(root) and bounds[0]<float(np.real(root))<bounds[1]:points.append(float(np.real(root)))
        if np.any(derivative(points)<=0) or not np.isfinite(derivative(points)).all():raise ValueError('时深多项式在输入数据的有效时间范围内必须严格单调递增')

    def transform(self,seconds,*,input_convention:str,require_confirmed:bool=True,valid_mask=None):
        if input_convention!=self.config['time_convention']:raise ValueError('输入时间与时深关系的单/双程时间类型不一致')
        times=np.asarray(seconds,dtype=float)/self.seconds_per_unit
        valid=np.ones(times.shape,dtype=bool) if valid_mask is None else np.broadcast_to(np.asarray(valid_mask,dtype=bool),times.shape).copy()
        if not np.isfinite(times[valid]).all():raise ValueError('输入时间包含非有限值')
        valid &= times>=0
        depth=np.full(times.shape,np.nan)
        if not valid.any():return depth
        bounds=np.array([times[valid].min(),times[valid].max()])
        if self.observed_bounds is not None:bounds=np.array([min(bounds[0],self.observed_bounds[0]),max(bounds[1],self.observed_bounds[1])])
        if self.config['method']=='polynomial':
            self.check_polynomial(bounds)
            with np.errstate(over='ignore',invalid='ignore'):depth[valid]=self.polynomial(times[valid])
        else:
            selected=times[valid].copy();tolerance=max(1.,abs(self.times[0]),abs(self.times[-1]))*1e-12
            selected[np.abs(selected-self.times[0])<=tolerance]=self.times[0]
            selected[np.abs(selected-self.times[-1])<=tolerance]=self.times[-1]
            depth[valid]=np.interp(selected,self.times,self.depths,left=np.nan,right=np.nan)
        self.observed_bounds=bounds
        depth*=self.depth_factor
        return np.where(np.isfinite(depth),depth,np.nan)

    def inverse(self,depth_m):
        """Align well depths within the actual loaded seismic time grid."""
        depth=np.asarray(depth_m,dtype=float)/self.depth_factor
        if self.config['method']=='table':
            result=np.interp(depth,self.depths,self.times,left=np.nan,right=np.nan)
            if self.bounds[0]>self.bounds[1]:return np.full(depth.shape,np.nan)
            low,high=np.interp(self.bounds,self.times,self.depths)
        else:
            bounds=self.bounds
            if bounds is None:
                finite=depth[np.isfinite(depth)]
                if not len(finite):return np.full(depth.shape,np.nan)
                upper=1.
                for _ in range(60):
                    if self.polynomial(upper)>=finite.max():break
                    upper*=2
                bounds=np.array([0.,upper])
            self.check_polynomial(bounds);low,high=self.polynomial(bounds)
            left=np.full(depth.shape,bounds[0]);right=np.full(depth.shape,bounds[1])
            for _ in range(52):
                middle=(left+right)/2;below=self.polynomial(middle)<depth
                left=np.where(below,middle,left);right=np.where(below,right,middle)
            result=(left+right)/2
        return np.where(np.isfinite(depth)&(depth>=low)&(depth<=high),result*self.seconds_per_unit,np.nan)
