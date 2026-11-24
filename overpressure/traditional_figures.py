"""Traditional-method figures shared by the worker exports and central plot windows."""
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from .charts import FigureStyle, apply_typography
from .well_points import gap_limit


def _curve_segments(frame):
    """Preserve missing rows and break lines across unsampled depth intervals."""
    shown = frame.copy()
    gaps = np.flatnonzero(np.diff(shown.Depth.to_numpy(dtype=float)) > gap_limit(shown.Depth)) + 1
    if not len(gaps):return shown
    # A NaN row before each disconnected interval breaks all plotted tracks.
    indices = np.arange(len(shown), dtype=float)
    shown.index = indices
    breaks = pd.DataFrame(np.nan, index=indices[gaps] - .5, columns=shown.columns)
    return pd.concat([shown, breaks]).sort_index()


def trend_figure(frame, trend, well, style=None):
    style = style or FigureStyle()
    figure = Figure(figsize=(9, 6), layout='constrained')
    ax = figure.subplots()
    valid = frame.loc[np.isfinite(frame.AC) & (frame.AC > 0)]
    shown = valid.iloc[::max(1, int(np.ceil(len(valid) / 20000)))]
    ax.scatter(shown.Depth, np.log(shown.AC), s=5, color='#7797ad', alpha=.65, label='AC data')
    depths = np.linspace(float(valid.Depth.min()), float(valid.Depth.max()), 200)
    ax.plot(depths, trend['a'] * depths + trend['b'], color='#b56524', linewidth=1.7, label='Normal compaction trend')
    points = np.asarray(trend['points'])
    ax.plot(points[:, 0], points[:, 1], 'o', color='#b56524', markersize=6, label='User calibration points')
    ax.set(xlabel='Depth (m)', ylabel='LN(AC [us/ft])', title=f'{well} · Normal compaction trend')
    ax.text(.02, .02, f"LN(AC) = {trend['a']:.9g} × Depth + {trend['b']:.9g}\nTwo-point calibration; not automatic regression", transform=ax.transAxes, fontsize=9, va='bottom')
    ax.legend(loc='best')
    if style.grid:ax.grid(True, alpha=.2)
    return apply_typography(figure, style)


def eaton_figure(frame, observations, well, style=None):
    style = style or FigureStyle(width_cm=28., height_cm=20.)
    figure = Figure(figsize=(11, 7), layout='constrained')
    axes = figure.subplots(1, 3, sharey=True)
    shown = _curve_segments(frame)
    axes[0].plot(shown.AC.where(shown.AC>0), shown.Depth, color='#6c96b0', linewidth=.8, label='AC')
    axes[0].plot(shown.AC_normal, shown.Depth, color='#b56524', linewidth=1.3, label='Normal AC')
    axes[0].set(xlabel='AC (us/ft)', ylabel='Depth (m)', title='Compaction trend')
    axes[1].plot(shown.P_h, shown.Depth, '--', color='#5794a0', linewidth=1, label='Hydrostatic')
    axes[1].plot(shown.P_o, shown.Depth, '--', color='#8d8981', linewidth=1, label='Overburden')
    axes[1].plot(shown.Pp_eaton, shown.Depth, color='#246c91', linewidth=1.2, label='Eaton pressure')
    if observations is not None and not observations.empty:
        finite = np.isfinite(observations.Pp) & np.isfinite(observations.Depth) & (observations.Pp >= 0) & (observations.Depth >= 0)
        matched = observations.supervised_valid.fillna(False).astype(bool) if 'supervised_valid' in observations else finite
        for mask, marker, label in [(finite & matched, 'o', 'Matched references'), (finite & ~matched, 'x', 'Unmatched references')]:
            if mask.any():axes[1].scatter(observations.loc[mask, 'Pp'], observations.loc[mask, 'Depth'], s=24, marker=marker, color='#b55337', label=label, zorder=4)
    axes[1].set(xlabel='Pressure (MPa)', title='Eaton prediction')
    axes[2].plot(shown.pressure_coefficient, shown.Depth, color='#246c91', linewidth=1.2, label='Pp / Ph')
    flagged = shown.quality_flags.fillna('').ne('') & np.isfinite(shown.pressure_coefficient)
    axes[2].scatter(shown.loc[flagged, 'pressure_coefficient'], shown.loc[flagged, 'Depth'], s=7, color='#b56524', label='Flagged points')
    axes[2].axvline(1., linestyle='--', color='#8d8981', linewidth=.8)
    axes[2].set(xlabel='Pressure coefficient (Pp / Ph)', title='Pressure coefficient')
    axes[0].invert_yaxis()
    for ax in axes:
        ax.legend(loc='best')
        if style.grid:ax.grid(True, alpha=.2)
    figure.suptitle(f'{well} · Eaton pressure prediction')
    return apply_typography(figure, style)
