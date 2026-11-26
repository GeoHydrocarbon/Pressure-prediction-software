"""Rendering is separate from overlay data retrieval and object controls."""
import numpy as np


def well_key(name):return 'well:'+name
def curve_key(ident):return 'curve:'+ident


def default_style(kind,mode='line'):
    return dict(visible=kind=='well',color='#243b53' if kind=='well' else '#c24f31',linewidth=1.5,
                linestyle='-',alpha=1.,mode=mode,marker='o',markersize=4.,width=8.,scale=1.,offset=0.,
                auto_range=True,minimum=0.,maximum=1.,show_name=True,show_track=True)


def selected_curves(settings,wells):
    styles=settings.get('styles',{})
    return [ref['id'] for well in wells if styles.get(well_key(well['name']),{}).get('visible',True)
            for ref in well.get('curves',[]) if styles.get(curve_key(ref['id']),{}).get('visible',False)]


def render_overlays(ax,wells,settings,time_range):
    x_limits,y_limits=ax.get_xlim(),ax.get_ylim()
    styles=settings.get('styles',{});tolerance=max(1e-8,float(settings.get('distance',25.)))
    low,high=sorted(time_range);span=abs(ax.get_xlim()[1]-ax.get_xlim()[0]);artists=[];legend=[]
    def line_style(style):return dict(color=style['color'],linewidth=style['linewidth'],linestyle=style['linestyle'],alpha=style['alpha'],zorder=6)
    for well in wells:
        ws=dict(default_style('well'),**styles.get(well_key(well['name']),{}))
        if not ws['visible']:continue
        distances=[d for d in (well.get('head_distance'),well.get('geometry_distance')) if d is not None]
        if not distances or min(distances)>tolerance:continue
        track=well.get('track');anchor=None
        if track is not None:
            x=np.asarray(track['x'],float);time=np.asarray(track['time'],float);d=np.asarray(track['distance'],float)
            valid=np.isfinite(x)&np.isfinite(time)&(time>=low-1e-7)&(time<=high+1e-7)
            time=np.clip(time,low,high)
            x=np.where(valid,x,np.nan);time=np.where(valid,time,np.nan)
            if valid.any():
                anchor=(x[np.flatnonzero(valid)[0]],time[np.flatnonzero(valid)[0]])
                if ws['show_track']:artists+=ax.plot(x,time,**line_style(ws))
        elif well.get('head_x') is not None:
            anchor=(well['head_x'],low)
            if ws['show_track']:artists+=ax.plot([well['head_x']]*2,[low,high],**line_style(ws))
        if anchor and ws['show_name']:
            below=anchor[1]<(low+high)/2
            artists.append(ax.annotate(well['name'],anchor,xytext=(4,-6 if below else 4),va='top' if below else 'bottom',textcoords='offset points',color=ws['color'],alpha=ws['alpha'],fontsize=9,clip_on=True,zorder=8))
        refs={ref['id']:ref for ref in well.get('curves',[])}
        for curve in well.get('curve_data',[]):
            ref=refs[curve['id']];style=dict(default_style('curve',ref.get('mode','line')),**styles.get(curve_key(curve['id']),{}))
            if not style['visible']:continue
            base=np.asarray(curve['x'],float);time=np.asarray(curve['time'],float);values=np.asarray(curve['values'],float);distance=np.asarray(curve['distance'],float)
            valid=np.isfinite(base)&np.isfinite(time)&np.isfinite(values)&(time>=low-1e-7)&(time<=high+1e-7)
            time=np.clip(time,low,high)
            if not valid.any():continue
            minimum,maximum=np.quantile(values[valid],[.01,.99]) if style['auto_range'] else (style['minimum'],style['maximum'])
            normalized=(values-minimum)/(maximum-minimum)-.5 if maximum>minimum else np.zeros_like(values)
            x=base+span*(normalized*style['width']*style['scale']+style['offset'])/100.
            x=np.where(valid,x,np.nan);time=np.where(valid,time,np.nan)
            label=well['name']+' / '+ref['name']+f' [{minimum:.4g}–{maximum:.4g}'+(' '+ref['unit'] if ref.get('unit') else '')+']'
            if style['mode'] in ('line','both'):
                item=ax.plot(x,time,label=label,**line_style(style))[0];artists.append(item);legend.append(item)
            if style['mode'] in ('points','both'):
                item=ax.scatter(x[valid],time[valid],s=style['markersize']**2,marker=style['marker'],color=style['color'],alpha=style['alpha'],zorder=7,label=label if style['mode']=='points' else None)
                artists.append(item)
                if style['mode']=='points':legend.append(item)
    if legend and settings.get('legend',True):ax.legend(handles=legend,fontsize=8,loc='best')
    ax.set_xlim(x_limits);ax.set_ylim(y_limits)
    return artists
