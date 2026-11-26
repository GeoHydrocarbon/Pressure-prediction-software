"""Reusable figures for desktop display and standalone PNG exports."""
from pathlib import Path
import os
from dataclasses import dataclass, asdict
import json

import numpy as np

from .runtime import user_data_root
os.environ.setdefault("MPLCONFIGDIR", str(user_data_root() / "outputs" / "matplotlib_cache"))
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.ticker import MaxNLocator
from matplotlib.text import Text
from matplotlib import rc_context
from .evaluation import paired_points, metrics
from .well_points import gap_limit


@dataclass
class FigureStyle:
    font_family: str = "Arial"
    font_size: float = 11
    title_size: float = 14
    show_title: bool = True
    title: str = ""
    width_cm: float = 18
    height_cm: float = 13
    dpi: int = 300
    point_size: float = 12
    alpha: float = .5
    grid: bool = True
    axis_label_size: float | None = None
    tick_size: float | None = None
    legend_size: float | None = None

    def validate(self):
        for name, low, high in (("font_size", 5, 36), ("title_size", 5, 48), ("width_cm", 5, 50),
                                ("height_cm", 5, 50), ("dpi", 72, 600), ("point_size", 1, 100), ("alpha", .05, 1)):
            value = getattr(self, name)
            if not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f"绘图设置 {name} 必须在 {low} 至 {high} 之间")
        for name in ('axis_label_size','tick_size','legend_size'):
            value=getattr(self,name)
            if value is not None and (not np.isfinite(value) or not 5<=value<=36):raise ValueError(f'绘图设置 {name} 必须在 5 至 36 之间')
        if not self.font_family.strip(): raise ValueError("字体不能为空")
        if (self.width_cm / 2.54 * self.dpi) * (self.height_cm / 2.54 * self.dpi) > 40_000_000:
            raise ValueError("导出图片超过 4000 万像素，请降低尺寸或 DPI")
        return self

    def save(self, path):
        self.validate(); path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        try: return cls(**json.loads(Path(path).read_text(encoding="utf-8"))).validate()
        except (OSError, ValueError, TypeError): return cls()


PALETTE = ["#236889", "#c18555", "#58958b", "#8c80a9", "#a98f43", "#6d899e"]


def figure_shell(style=None):
    style = (style or FigureStyle()).validate()
    return Figure(figsize=(style.width_cm / 2.54, style.height_cm / 2.54), dpi=100, layout="constrained")


def apply_typography(figure,style,title_axes=None):
    """Shared typography; custom titles apply only to plot axes, not colorbars."""
    for text in figure.findobj(Text):
        text.set_fontfamily([style.font_family,'Microsoft YaHei','DejaVu Sans'])
        text.set_fontsize(style.font_size);text.set_color('#263c4a')
    for ax in figure.axes:
        ax.xaxis.label.set_fontsize(style.axis_label_size or style.font_size)
        ax.yaxis.label.set_fontsize(style.axis_label_size or style.font_size)
        ax.tick_params(labelsize=style.tick_size or style.font_size)
        for axis in (ax.xaxis,ax.yaxis):axis.get_offset_text().set_fontsize(style.tick_size or style.font_size)
        legend=ax.get_legend()
        if legend:
            for text in legend.get_texts():text.set_fontsize(style.legend_size or style.font_size)
    for ax in figure.axes if title_axes is None else title_axes:
        title=(style.title.strip() or ax.get_title()) if style.show_title else ''
        ax.set_title(title,pad=14,fontsize=style.title_size,fontweight=700)
    return figure


def finish_style(figure, style=None):
    style = style or FigureStyle()
    for ax in figure.axes:
        ax.spines[["top", "right"]].set_visible(False)
        for spine in ax.spines.values(): spine.set_color("#87939d"); spine.set_linewidth(.7)
        ax.tick_params(colors="#344550", width=.7, length=3)
        ax.set_axisbelow(True)
        ax.grid(style.grid, color="#e7edf0", linewidth=.6) if style.grid else ax.grid(False)
        legend = ax.get_legend()
        if legend: legend.set_frame_on(False)
    return apply_typography(figure,style)


def training_figure(history, style=None):
    figure = figure_shell(style)
    ax = figure.subplots()
    ax.plot(history["epoch"], history["train_data"], label="Training data MSE", color="#23628b")
    if "validation_data" in history:
        ax.plot(history["epoch"], history["validation_data"], label="Validation data MSE", color="#c18555", linewidth=1.6)
    unit=history['iteration_unit'].iloc[0] if 'iteration_unit' in history else 'epochs'
    ax.set(xlabel={'trees':'Number of trees','boosting_rounds':'Boosting round'}.get(unit,'Epoch'), ylabel="MSE (MPa squared)", title="Training and internal validation" if "validation_data" in history else "Final refit: fixed iterations, no validation or blind wells")
    ax.grid(alpha=.2); ax.legend()
    return finish_style(figure, style)


def pressure_figure(frame, well, style=None, observations=None, max_gap_m=0, source_label=None):
    frame = frame.loc[frame.well.astype(str) == str(well)].sort_values("Depth", kind="stable")
    if frame.empty:
        raise ValueError("未找到该井的压力数据")
    figure = figure_shell(style); ax = figure.subplots()
    depth = frame.Depth.to_numpy(dtype=float)
    differences = np.diff(depth); positive = differences[np.isfinite(differences) & (differences > 0)]
    breaks = np.flatnonzero(differences > gap_limit(depth, max_gap_m)) + 1 if len(positive) else np.array([], dtype=int)
    for field, label, color in (("Pp", "Reference", "#c47b31"), ("Pp_pred", "Prediction", "#23628b")):
        if observations is not None and field == "Pp": continue
        if field in frame:
            values = frame[field].to_numpy(dtype=float)
            ax.plot(np.insert(values, breaks, np.nan), np.insert(depth, breaks, np.nan), label=label, color=color, linewidth=1.1)
    if observations is not None:
        points = observations.loc[observations.well.astype(str) == str(well)]
        valid = np.isfinite(points.Depth) & (points.Depth >= 0) & np.isfinite(points.Pp) & (points.Pp >= 0)
        matched = points.match_status.isin(["matched_exact", "matched_interpolated"])
        shown = points.loc[valid & matched]
        if len(shown): ax.scatter(shown.Pp, shown.Depth, marker="o", s=38, facecolor="#c77d45", edgecolor="white", linewidth=.8, zorder=4, label="Measured pressure")
        unmatched = points.loc[valid & ~matched]
        if len(unmatched): ax.scatter(unmatched.Pp, unmatched.Depth, marker="x", s=38, color="#9a6266", linewidth=1.2, zorder=4, label="Unmatched measurement")
    ax.invert_yaxis(); ax.set(xlabel="Pressure (MPa)", ylabel="Depth (m)", title=str(well))
    ax.grid(alpha=.2); ax.legend()
    if source_label: figure.supxlabel(source_label)
    return finish_style(figure, style)


def folds_figure(frame, style=None):
    figure = figure_shell(style); ax = figure.subplots()
    positions = np.arange(len(frame))
    ax.bar(positions - .18, frame.mae, .36, label="MAE")
    ax.bar(positions + .18, frame.rmse, .36, label="RMSE")
    ax.set_xticks(positions, frame.validation_well.astype(str))
    ax.set(ylabel="Error (MPa)", title="Leave-one-development-well-out validation")
    ax.legend(); ax.grid(axis="y", alpha=.2)
    return finish_style(figure, style)


def scatter_figure(frame, well=None, style=None, source_label="Evaluation"):
    style = style or FigureStyle()
    selected = frame if well is None else frame.loc[frame.well.astype(str) == str(well)]
    pairs = paired_points(selected)
    figure = figure_shell(style); ax = figure.subplots()
    if not pairs.empty:
        groups = list(pairs.groupby("well", sort=False)) if "well" in pairs else [("Points", pairs)]
        for i, (name, group) in enumerate(groups):
            ax.scatter(group.Pp, group.Pp_pred, s=style.point_size, alpha=style.alpha, color=PALETTE[i % len(PALETTE)],
                       edgecolors="none", label=str(name), rasterized=True)
        # Match axis ranges to make bias visible without adding a reference line.
        low = float(min(pairs.Pp.min(), pairs.Pp_pred.min())); high = float(max(pairs.Pp.max(), pairs.Pp_pred.max()))
        pad = max((high - low) * .06, .5); ax.set_xlim(low - pad, high + pad); ax.set_ylim(low - pad, high + pad)
        ax.set_aspect("equal", adjustable="box")
        result = metrics(pairs.Pp, pairs.Pp_pred)
        r2 = f"{result['r2']:.3f}" if result["r2"] is not None else "N/A"
        ax.text(.97, .05, f"R² = {r2}\nn = {result['count']:,}", transform=ax.transAxes, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#dbe3e8", alpha=.94))
        if len(groups) > 1: ax.legend(loc="upper left", ncol=min(3, len(groups)), markerscale=1.5)
    else:
        ax.text(.5, .5, "No valid reference/prediction pairs", ha="center", va="center", transform=ax.transAxes)
    ax.set(xlabel="Reference pressure (MPa)", ylabel="Predicted pressure (MPa)", title="Prediction vs reference" + (f" · {well}" if well else ""))
    figure.supxlabel(source_label)
    return finish_style(figure, style)


def export_figure(figure, path, style=None):
    style = (style or FigureStyle()).validate()
    size = figure.get_size_inches().copy()
    try:
        figure.set_size_inches(style.width_cm / 2.54, style.height_cm / 2.54, forward=False)
        with rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
            figure.savefig(path, dpi=style.dpi, facecolor="white")
    finally:
        figure.set_size_inches(size, forward=False)


def seismic_figure(section, limits, *, figure=None, cmap="viridis", xlim=None, ylim=None, overlay=None, overlay_color="#e35050", overlay_name="Horizon", well=None, well_label="", well_width=3,style=None,overlays=None):
    if figure is None: figure = Figure(figsize=(8, 5), layout="constrained")
    else: figure.clear()
    ax = figure.subplots()
    def edges(values):
        values = np.asarray(values, dtype=float)
        if len(values) == 1: return np.array([values[0] - .5, values[0] + .5])
        return np.r_[values[0] - (values[1] - values[0]) / 2, (values[:-1] + values[1:]) / 2, values[-1] + (values[-1] - values[-2]) / 2]
    low, high = limits
    if low == high: low, high = low - max(1., abs(low) * .01), high + max(1., abs(high) * .01)
    ax.set_facecolor("#eeeeee")
    image = ax.pcolormesh(edges(section.x), edges(section.y), np.ma.masked_invalid(section.values), shading="flat", cmap=cmap, vmin=low, vmax=high)
    if section.invert_y: ax.invert_yaxis()
    ax.set(xlabel=section.xlabel, ylabel=section.ylabel, title=section.title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if section.ylabel == "Inline": ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.ticklabel_format(useOffset=False, style="plain")
    if xlim is not None: ax.set_xlim(*xlim)
    if ylim is not None: ax.set_ylim(*(tuple(reversed(ylim)) if section.invert_y else ylim))
    if overlays is not None:
        for layer in overlays:
            times=layer['time']
            if len(times)==len(section.x) and np.isfinite(times).any():
                ax.plot(section.x,times,color=layer['color'],linewidth=layer['width'],label=layer['label'])
        if ax.lines:ax.legend(loc='upper right',framealpha=.85)
    elif overlay is not None and len(overlay) == len(section.x) and np.isfinite(overlay).any():
        ax.plot(section.x, overlay, color=overlay_color, linewidth=1.6, marker=".", markersize=2.5, label=overlay_name)
        ax.legend(loc="upper right", framealpha=.85)
    if well and np.isfinite(well['center']).any():
        center, time = well['center'], well['time']
        combined = np.r_[well['curve'],well['seismic']]; finite = combined[np.isfinite(combined)]
        ax.plot(center,time,color='#17212b',linewidth=.8,linestyle='--',label=well_label+' trajectory')
        if len(finite):
            low_curve, high_curve = finite.min(), finite.max(); span = high_curve-low_curve or 1
            ax.plot(center+well_width*(well['curve']-low_curve)/span,time,color='#fff9bf',linewidth=2.5)
            ax.plot(center+well_width*(well['curve']-low_curve)/span,time,color='#d92344',linewidth=1.2,
                    label=f'{well_label} [{low_curve:.3g}, {high_curve:.3g}]')
            if np.isfinite(well['seismic']).any():
                ax.plot(center+well_width*(well['seismic']-low_curve)/span,time,color='#00d4ed',linewidth=1.3,label='Seismic pressure at well (MPa)')
        ax.legend(loc='upper right',framealpha=.9,fontsize=8)
    figure.colorbar(image, ax=ax, label="Pressure (MPa)")
    if not np.isfinite(section.values).any():
        ax.text(.5, .5, "No valid points on this slice", ha="center", va="center", transform=ax.transAxes)
    style=style or FigureStyle(grid=False)
    ax.set_axisbelow(False)
    if style.grid:ax.grid(True,color='#d6e1e8',linewidth=.5,alpha=.55)
    else:ax.grid(False)
    return apply_typography(figure,style,title_axes=[ax])


def save_figure(figure, path):
    FigureCanvasAgg(figure)
    figure.savefig(path, dpi=160)
    figure.clear()
