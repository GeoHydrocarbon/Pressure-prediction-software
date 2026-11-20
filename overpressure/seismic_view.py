"""Bounded SEG-Y result reader and geometry-aware, non-interpolating slices."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass
class Slice:
    values: np.ndarray
    x: np.ndarray
    y: np.ndarray
    xlabel: str
    ylabel: str
    title: str
    invert_y: bool


def bounded_axis(values):
    low, high = int(np.min(values)), int(np.max(values))
    if high - low + 1 > 4096:
        raise ValueError("当前小范围浏览器的线号跨度不能超过 4096，请选择更紧凑的范围")
    return np.arange(low, high + 1, dtype=np.int64)


class SeismicView:
    def __init__(self, pressure, inline, crossline, delay_ms, interval_ms):
        self.purpose = "unknown"
        self.pressure = np.asarray(pressure, dtype=np.float32)
        self.inline = np.asarray(inline); self.crossline = np.asarray(crossline)
        self.delay_ms = np.asarray(delay_ms, dtype=float)
        if self.pressure.ndim != 2 or not 1 <= len(self.pressure) <= 512 or self.pressure.shape[1] < 2 or self.pressure.size > 2_000_000:
            raise ValueError("当前浏览器支持 1–512 道、每道至少两个采样点的小范围成果")
        n, ns = self.pressure.shape
        if any(v.shape != (n,) for v in (self.inline, self.crossline, self.delay_ms)):
            raise ValueError("切片几何数组与压力道数不一致")
        if any(not np.isfinite(v).all() for v in (self.inline, self.crossline, self.delay_ms)) or not np.isfinite(interval_ms) or interval_ms <= 0:
            raise ValueError("切片坐标或采样间隔无效")
        if not np.equal(self.inline, self.inline.astype(np.int64)).all() or not np.equal(self.crossline, self.crossline.astype(np.int64)).all():
            raise ValueError("Inline/Crossline 必须为整数线号")
        self.inline = self.inline.astype(np.int64); self.crossline = self.crossline.astype(np.int64)
        if len(np.unique(np.stack((self.inline, self.crossline), axis=1), axis=0)) != n:
            raise ValueError("成果中存在重复 Inline/Crossline，不能唯一定位切片")
        if np.isinf(self.pressure).any() or not np.isfinite(self.pressure).any():
            raise ValueError("压力成果包含无穷值或没有有效点")
        self.interval_ms = float(interval_ms)
        self.t0_ms = float(self.delay_ms.min())
        offsets = (self.delay_ms - self.t0_ms) / self.interval_ms
        if not np.allclose(offsets, np.round(offsets), atol=1e-7, rtol=0):
            raise ValueError("各道起始时间无法对齐共同采样网格，请先统一时间采样")
        self.offsets = np.round(offsets).astype(np.int64)
        nt = int(self.offsets.max()) + ns
        self.xlines = bounded_axis(self.crossline); self.ilines = bounded_axis(self.inline)
        if nt * max(len(self.xlines), len(self.ilines)) > 2_000_000 or len(self.xlines) * len(self.ilines) > 2_000_000:
            raise ValueError("切片网格超过当前小范围浏览器上限，请缩小范围")
        self.times_ms = self.t0_ms + np.arange(nt) * self.interval_ms
        finite = self.pressure[np.isfinite(self.pressure)]
        self.limits = (float(finite.min()), float(finite.max()))

    @classmethod
    def from_cache(cls, path):
        with np.load(path, allow_pickle=False) as cache:
            view = cls(cache["pressure"], cache["inline"], cache["crossline"], cache["delay_ms"], float(cache["interval_ms"].item()))
            if "purpose" in cache: view.purpose = str(cache["purpose"].item())
            return view

    def section(self, mode, value):
        if mode == "time":
            index = int(value)
            if not 0 <= index < len(self.times_ms): raise ValueError("时间采样索引超出范围")
            values = np.full((len(self.ilines), len(self.xlines)), np.nan, dtype=np.float32)
            local = index - self.offsets
            for row in np.flatnonzero((local >= 0) & (local < self.pressure.shape[1])):
                values[self.inline[row] - self.ilines[0], self.crossline[row] - self.xlines[0]] = self.pressure[row, local[row]]
            return Slice(values, self.xlines, self.ilines, "Crossline", "Inline", f"Time slice {self.times_ms[index]:g} ms", False)
        if mode not in ("inline", "crossline"): raise ValueError("未知切片方向")
        fixed = self.inline if mode == "inline" else self.crossline
        variable = self.crossline if mode == "inline" else self.inline
        axis = self.xlines if mode == "inline" else self.ilines
        if int(value) not in fixed: raise ValueError("所选线号不在成果中")
        values = np.full((len(self.times_ms), len(axis)), np.nan, dtype=np.float32)
        for row in np.flatnonzero(fixed == int(value)):
            time = self.offsets[row] + np.arange(self.pressure.shape[1])
            values[time, variable[row] - axis[0]] = self.pressure[row]
        return Slice(values, axis, self.times_ms, "Crossline" if mode == "inline" else "Inline", "Time (ms)", f"{mode.title()} {int(value)}", True)


def prepare_view(result_folder):
    """Read completed pressure+mask through the worker's segyio installation."""
    import segyio
    from .data import sha256_file

    folder = Path(result_folder).resolve()
    report = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    if report.get("status") != "completed" or report.get("round_trip_verified") is not True:
        raise ValueError("仅能浏览已完成且通过重读校验的预测成果")
    if report.get("pressure_unit") != "MPa" or report.get("output_domain") != "input_time_grid":
        raise ValueError("成果压力单位或采样域不受支持")
    if not isinstance(report.get("shape"), list) or len(report["shape"]) != 2 or not 1 <= report["shape"][0] <= 512:
        raise ValueError("请选择最多 512 道的小范围成果")
    pressure_path, mask_path = folder / "pressure.sgy", folder / "validity.sgy"
    for path, key in ((pressure_path, "pressure_sha256"), (mask_path, "validity_sha256")):
        if key in report and sha256_file(path) != report[key]: raise ValueError("成果文件校验失败，文件可能已被修改")
    fields = report["project"]["seismic"]["header_fields"]
    with segyio.open(str(pressure_path), "r", ignore_geometry=True) as pressure, segyio.open(str(mask_path), "r", ignore_geometry=True) as mask:
        if pressure.tracecount != report["shape"][0] or len(pressure.samples) != report["shape"][1] or pressure.tracecount != mask.tracecount or len(pressure.samples) != len(mask.samples):
            raise ValueError("压力、掩码或成果记录的尺寸不一致")
        if pressure.tracecount * len(pressure.samples) > 2_000_000:
            raise ValueError("当前切片浏览器最多读取 200 万采样点，请缩小范围")
        if int(pressure.format) != 5 or int(mask.format) != 5: raise ValueError("成果必须采用 IEEE float32 格式")
        n = pressure.tracecount
        data = np.stack([pressure.trace[i].copy() for i in range(n)])
        valid = np.stack([mask.trace[i].copy() for i in range(n)])
        if not np.isin(valid, [0, 1]).all() or not np.array_equal(np.isfinite(data), valid.astype(bool)) or np.isinf(data).any():
            raise ValueError("压力无效区与独立掩码不一致")
        arrays = {}
        for key, field in {**fields, "delay_ms": 109, "interval_us": 117, "sample_count": 115}.items():
            values = np.asarray(pressure.attributes(field)[:]).copy()
            if not np.array_equal(values, np.asarray(mask.attributes(field)[:])):
                raise ValueError("压力与掩码几何或时间道头不一致")
            arrays[key] = values
        interval_us = int(pressure.bin[segyio.BinField.Interval])
        if interval_us <= 0 or mask.bin[segyio.BinField.Interval] != interval_us or np.any(arrays["interval_us"] != interval_us) or np.any(arrays["sample_count"] != data.shape[1]):
            raise ValueError("成果采样间隔或采样点数不一致")
    SeismicView(data, arrays["inline"], arrays["crossline"], arrays["delay_ms"], interval_us / 1000.)
    target = folder / "view.npz"; partial = folder / "view.partial.npz"
    np.savez_compressed(partial, pressure=data, inline=arrays["inline"], crossline=arrays["crossline"], delay_ms=arrays["delay_ms"], interval_ms=np.array(interval_us / 1000.), purpose=np.array(report.get("purpose", "unknown")))
    partial.replace(target)
    return target
