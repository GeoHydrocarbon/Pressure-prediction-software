"""Minimum curvature: MD, inclination from vertical, clockwise azimuth from north.

Equations: https://romanshor.github.io/Introduction-to-Drilling-Engineering/main/chapters/chapter12_dd_calculations.html
East is X, North is Y; all returned distances are metres from the wellhead.
"""
import numpy as np


def minimum_curvature(md, inclination, azimuth, *, angle_unit='deg', initial_segment='require_zero'):
    md=np.asarray(md,dtype=float);inc=np.asarray(inclination,dtype=float);azi=np.asarray(azimuth,dtype=float)
    if md.ndim!=1 or not len(md) or inc.shape!=md.shape or azi.shape!=md.shape: raise ValueError('测斜数据长度不一致或为空')
    if not np.isfinite(np.column_stack([md,inc,azi])).all(): raise ValueError('MD、井斜角、方位角含空值或非数值，不能跨缺失测点累计轨迹')
    if (md<0).any() or (np.diff(md)<=0).any(): raise ValueError('每口井 MD 必须非负且严格递增，不能重复或倒序')
    if angle_unit not in ('deg','rad'): raise ValueError('角度单位必须为 deg 或 rad')
    limit=np.pi if angle_unit=='rad' else 180.
    if ((inc<0)|(inc>limit)).any(): raise ValueError('井斜角必须在 0～180°（或 0～π rad）之间，0 为垂直向下')
    if ((azi<0)|(azi>2*limit)).any(): raise ValueError('方位角必须在 0～360°（或 0～2π rad）之间')
    if initial_segment not in ('require_zero','first_direction'): raise ValueError('未知首段处理方式')
    if angle_unit=='deg': inc=np.deg2rad(inc);azi=np.deg2rad(azi)
    prepend=md[0]>1e-8
    if prepend:
        if initial_segment=='require_zero': raise ValueError('首个测点 MD 不为 0；请补充起点，或明确选择“首段沿首测点方向”')
        md=np.r_[0.,md];inc=np.r_[inc[0],inc];azi=np.r_[azi[0],azi]
    directions=np.column_stack([np.sin(inc)*np.sin(azi),np.sin(inc)*np.cos(azi),np.cos(inc)])
    dogleg=np.arccos(np.clip(np.sum(directions[1:]*directions[:-1],axis=1),-1.,1.))
    if (dogleg>=np.pi-1e-7).any(): raise ValueError('相邻测点方向接近反向，最小曲率解不唯一，请核对角度')
    ratio=np.ones_like(dogleg);valid=dogleg>1e-8;ratio[valid]=2*np.tan(dogleg[valid]/2)/dogleg[valid]
    displacement=(np.diff(md)*ratio/2)[:,None]*(directions[1:]+directions[:-1])
    result=np.vstack([np.zeros(3),np.cumsum(displacement,axis=0)])
    if not np.isfinite(result).all(): raise ValueError('轨迹计算产生非有限值，请核对测斜数据')
    return result[1:] if prepend else result


def survey_positions(frame, asset):
    frame=frame.copy()
    if frame.well.isna().any(): raise ValueError('测斜数据含空井名，请指定井名列或关联井')
    frame[['x','y','tvd']]=np.nan
    for name,group in frame.groupby('well',sort=False):
        try:
            xyz=minimum_curvature(group.depth,group.inclination,group.azimuth,
                angle_unit=asset.get('angle_unit','deg'),initial_segment=asset.get('initial_segment','require_zero'))
        except ValueError as exc: raise ValueError(f'{name}: {exc}') from exc
        frame.loc[group.index,['x','y','tvd']]=xyz
    return frame
