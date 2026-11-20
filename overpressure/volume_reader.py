"""Streaming integrity checks, compact geometry index, and bounded slice reads."""
from contextlib import ExitStack
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from .jobs import check_cancel, checked_sha256, emit, save_json
from .seismic_view import Slice


MAX_DISPLAY_CELLS = 500_000


def stamp(path):
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def prepare_index(result_folder, cancel_file=None):
    import segyio
    folder = Path(result_folder).resolve(); report_path = folder / "result.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "completed" or report.get("round_trip_verified") is not True:
        raise ValueError("仅能浏览已完成且通过重读校验的预测成果")
    if report.get("pressure_unit") != "MPa" or report.get("output_domain") != "input_time_grid": raise ValueError("成果单位或采样域不受支持")
    paths = {"pressure": folder / "pressure.sgy", "validity": folder / "validity.sgy"}
    signatures = {key: stamp(path) for key, path in paths.items()}; report_signature = stamp(report_path)
    hashes = {}
    for key, path in paths.items():
        emit("stage", message=f"正在校验 {path.name} 完整文件")
        hashes[key] = checked_sha256(path, cancel_file)
        if report.get(f"{key}_sha256") and hashes[key] != report[f"{key}_sha256"]:
            raise ValueError("成果文件校验失败，文件可能已被修改")
    fields = {**report["project"]["seismic"]["header_fields"], "delay_ms": 109, "interval_us": 117, "sample_count": 115}
    with ExitStack() as stack:
        pressure, mask = [stack.enter_context(segyio.open(str(paths[key]), "r", ignore_geometry=True)) for key in ("pressure", "validity")]
        n, ns = pressure.tracecount, len(pressure.samples)
        if not 1 <= n <= 2_000_000 or ns < 2: raise ValueError("当前几何索引支持最多 200 万道且每道至少两个采样点")
        if report.get("shape") != [n, ns] or mask.tracecount != n or len(mask.samples) != ns or any(int(f.format) != 5 for f in (pressure, mask)):
            raise ValueError("压力、掩码或记录的结构不一致")
        geometry = {}
        for key, field in fields.items():
            check_cancel(cancel_file)
            a = np.asarray(pressure.attributes(field)[:]).copy(); b = np.asarray(mask.attributes(field)[:])
            if not np.array_equal(a, b): raise ValueError("压力与掩码的几何或时间道头不一致")
            geometry[key] = a
        dt_us = int(pressure.bin[segyio.BinField.Interval])
        if dt_us <= 0 or mask.bin[segyio.BinField.Interval] != dt_us or np.any(geometry["interval_us"] != dt_us) or np.any(geometry["sample_count"] != ns):
            raise ValueError("成果采样间隔或采样点数不一致")
        pair = np.stack((geometry["inline"], geometry["crossline"]), axis=1)
        if len(np.unique(pair, axis=0)) != n: raise ValueError("成果中存在重复 Inline/Crossline")
        offsets = (geometry["delay_ms"] - geometry["delay_ms"].min()) * 1000. / dt_us
        if not np.allclose(offsets, np.round(offsets), atol=1e-7, rtol=0): raise ValueError("各道起始时间无法对齐共同采样网格")
        offsets = np.round(offsets).astype(np.int64); nt = ns + int(offsets.max())
        if nt > 1_000_000: raise ValueError("共同时间轴超过当前 100 万采样点上限，请统一起始时间")
        for key in ("inline", "crossline"):
            if int(geometry[key].max()) - int(geometry[key].min()) + 1 > 1_000_000: raise ValueError("线号跨度超过当前浏览范围上限")
        minimum, maximum, valid_points = float("inf"), -float("inf"), 0
        block = max(1, min(32, 262144 // ns))
        for start in range(0, n, block):
            check_cancel(cancel_file)
            rows = range(start, min(start + block, n))
            p = np.stack([pressure.trace[i].copy() for i in rows]); m = np.stack([mask.trace[i].copy() for i in rows])
            if np.isinf(p).any() or not np.isin(m, [0, 1]).all() or not np.array_equal(np.isfinite(p), m.astype(bool)):
                raise ValueError("压力无效区与掩码不一致")
            values = p[np.isfinite(p)]
            if len(values): minimum = min(minimum, float(values.min())); maximum = max(maximum, float(values.max())); valid_points += len(values)
            emit("seismic_progress", completed=min(start + block, n), total=n, phase="index")
        if not valid_points: raise ValueError("成果中没有有效压力值")
    check_cancel(cancel_file)
    if any(stamp(paths[k]) != signatures[k] for k in paths) or stamp(report_path) != report_signature:
        raise ValueError("建立索引期间成果发生变化，请重新载入")
    target = folder / "volume_index.npz"; partial = folder / "volume_index.partial.npz"
    np.savez_compressed(partial, inline=geometry["inline"], crossline=geometry["crossline"], offsets=offsets,
                        x=scaled_coordinates(geometry["x"], geometry["coordinate_scalar"]),
                        y=scaled_coordinates(geometry["y"], geometry["coordinate_scalar"]), coordinate_units=geometry["coordinate_units"],
                        time_convention=np.array(report["project"]["seismic"]["time_convention"]),
                        ns=np.array(ns), nt=np.array(nt), interval_ms=np.array(dt_us / 1000.), t0_ms=np.array(float(geometry["delay_ms"].min())),
                        limits=np.array([minimum, maximum]), purpose=np.array(report.get("purpose", "unknown")))
    partial.replace(target)
    save_json(folder / "volume_index.json", {"schema_version": 1, "status": "completed", "files": signatures,
              "report_sha256": checked_sha256(report_path, cancel_file), "index_sha256": checked_sha256(target, cancel_file),
              "verified_hashes": hashes, "valid_points": valid_points, "amplitudes_cached": False})
    return target


def require_fresh(index_path):
    path = Path(index_path); folder = path.parent
    meta = json.loads((folder / "volume_index.json").read_text(encoding="utf-8"))
    if meta.get("schema_version") != 1 or meta.get("status") != "completed": raise ValueError("几何索引尚未完成")
    for key, name in (("pressure", "pressure.sgy"), ("validity", "validity.sgy")):
        if stamp(folder / name) != meta["files"][key]: raise ValueError("成果文件发生变化，请重新载入并校验")
    if checked_sha256(folder / "result.json") != meta["report_sha256"]: raise ValueError("成果记录发生变化，请重新载入")
    return meta


@dataclass
class VolumeIndex:
    path: Path
    inline: np.ndarray
    crossline: np.ndarray
    offsets: np.ndarray
    ns: int
    nt: int
    interval_ms: float
    t0_ms: float
    limits: tuple
    purpose: str
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    coordinate_units: np.ndarray | None = None
    time_convention: str = "twt"

    @property
    def times_ms(self): return self.t0_ms + np.arange(self.nt) * self.interval_ms

    @classmethod
    def load(cls, path):
        path = Path(path).resolve(); meta = require_fresh(path)
        if checked_sha256(path) != meta["index_sha256"]: raise ValueError("几何索引校验失败，请重新载入")
        with np.load(path, allow_pickle=False) as data:
            return cls(path, data["inline"], data["crossline"], data["offsets"], int(data["ns"].item()), int(data["nt"].item()),
                       float(data["interval_ms"].item()), float(data["t0_ms"].item()), tuple(data["limits"].tolist()), str(data["purpose"].item()),
                       data["x"] if "x" in data else None, data["y"] if "y" in data else None,
                       data["coordinate_units"] if "coordinate_units" in data else None,
                       str(data["time_convention"].item()) if "time_convention" in data else json.loads((path.parent / "result.json").read_text(encoding="utf-8"))["project"]["seismic"]["time_convention"])


def scaled_coordinates(raw, scalar):
    scalar = np.asarray(scalar, dtype=float)
    factor = np.ones_like(scalar)
    factor[scalar > 0] = scalar[scalar > 0]
    factor[scalar < 0] = 1 / np.abs(scalar[scalar < 0])
    return np.asarray(raw, dtype=float) * factor


def sample_grid(nx, ny, limit):
    if not 1 <= limit <= MAX_DISPLAY_CELLS: raise ValueError("显示格数上限无效")
    sx = max(1, math.ceil(nx / min(1024, limit)))
    width = math.ceil(nx / sx)
    sy = max(1, math.ceil(ny / max(1, limit // width)))
    return sx, sy


class VolumeReader:
    """Read just the requested line/sample intervals; never materialize a cube."""
    def __init__(self, index_path):
        import segyio
        self.index = VolumeIndex.load(index_path); folder = self.index.path.parent
        self.pressure = segyio.open(str(folder / "pressure.sgy"), "r", ignore_geometry=True)
        try: self.mask = segyio.open(str(folder / "validity.sgy"), "r", ignore_geometry=True)
        except BaseException:
            self.pressure.close(); raise
        self.last_request = {}

    def close(self):
        self.pressure.close(); self.mask.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()

    def values(self, row, sample):
        p = np.asarray(self.pressure.trace[int(row), sample]).reshape(-1)
        m = np.asarray(self.mask.trace[int(row), sample]).reshape(-1)
        if np.isinf(p).any() or not np.isin(m, [0, 1]).all() or not np.array_equal(np.isfinite(p), m.astype(bool)):
            raise ValueError("当前切片压力与掩码不一致，请重新载入")
        self.last_request["sample_values_read"] += len(p)
        self.last_request["traces_read"] += 1
        return p

    def section(self, mode, value, max_cells=MAX_DISPLAY_CELLS):
        require_fresh(self.index.path); idx = self.index
        self.last_request = {"sample_values_read": 0, "traces_read": 0}
        if mode == "time":
            value = int(value)
            if not 0 <= value < idx.nt: raise ValueError("时间索引超出范围")
            xmin, xmax, ymin, ymax = int(idx.crossline.min()), int(idx.crossline.max()), int(idx.inline.min()), int(idx.inline.max())
            sx, sy = sample_grid(xmax - xmin + 1, ymax - ymin + 1, max_cells)
            x, y = np.arange(xmin, xmax + 1, sx), np.arange(ymin, ymax + 1, sy)
            values = np.full((len(y), len(x)), np.nan, dtype=np.float32)
            local = value - idx.offsets
            selected = (local >= 0) & (local < idx.ns) & ((idx.crossline - xmin) % sx == 0) & ((idx.inline - ymin) % sy == 0)
            for row in np.flatnonzero(selected): values[(idx.inline[row] - ymin) // sy, (idx.crossline[row] - xmin) // sx] = self.values(row, int(local[row]))[0]
            section = Slice(values, x, y, "Crossline", "Inline", f"Time slice {idx.t0_ms + value * idx.interval_ms:g} ms", False)
        else:
            if mode not in ("inline", "crossline"): raise ValueError("未知切片方向")
            fixed, variable = (idx.inline, idx.crossline) if mode == "inline" else (idx.crossline, idx.inline)
            value = int(value)
            if value not in fixed: raise ValueError("所选线号不在成果中")
            low, high = int(variable.min()), int(variable.max())
            sx, sy = sample_grid(high - low + 1, idx.nt, max_cells)
            x = np.arange(low, high + 1, sx); samples = np.arange(0, idx.nt, sy)
            y = idx.t0_ms + samples * idx.interval_ms; values = np.full((len(y), len(x)), np.nan, dtype=np.float32)
            for row in np.flatnonzero((fixed == value) & ((variable - low) % sx == 0)):
                local = samples - idx.offsets[row]; good = (local >= 0) & (local < idx.ns)
                if good.any(): values[good, (variable[row] - low) // sx] = self.values(row, slice(int(local[good][0]), int(local[good][-1]) + 1, sy))
            section = Slice(values, x, y, "Crossline" if mode == "inline" else "Inline", "Time (ms)", f"{mode.title()} {value}", True)
        self.last_request.update(x_stride=sx, y_stride=sy, output_cells=int(values.size))
        require_fresh(idx.path)
        return section

    def horizon_section(self, times_ms, name, method="linear", max_cells=MAX_DISPLAY_CELLS,point_types=None):
        require_fresh(self.index.path); idx = self.index
        if method not in ("linear", "nearest"): raise ValueError("层位时间取值方式无效")
        times_ms = np.asarray(times_ms, dtype=float)
        if times_ms.shape != idx.inline.shape: raise ValueError("层位与地震道数不一致")
        kinds=np.where(np.isfinite(times_ms),1,0).astype(np.uint8) if point_types is None else np.asarray(point_types)
        if kinds.shape!=times_ms.shape:raise ValueError('层位点来源标记与道数不一致')
        self.last_request = {"sample_values_read": 0, "traces_read": 0}
        covered=np.isfinite(times_ms)
        bounds=covered if covered.any() else np.ones(len(times_ms),dtype=bool)
        xmin, xmax, ymin, ymax = int(idx.crossline[bounds].min()), int(idx.crossline[bounds].max()), int(idx.inline[bounds].min()), int(idx.inline[bounds].max())
        sx, sy = sample_grid(xmax-xmin+1, ymax-ymin+1, max_cells)
        x, y = np.arange(xmin, xmax+1, sx), np.arange(ymin, ymax+1, sy)
        values = np.full((len(y), len(x)), np.nan, dtype=np.float32)
        self.last_point_types=np.zeros(values.shape,dtype=np.uint8)
        sample = (times_ms - idx.t0_ms) / idx.interval_ms - idx.offsets
        in_range=np.isfinite(sample) & (sample >= -1e-7) & (sample <= idx.ns-1+1e-7)
        selected=np.flatnonzero(in_range)
        if (sx>1 or sy>1) and len(selected):
            # Keep a real mapped trace in each occupied display cell. Requiring
            # modulo alignment can discard every sparse horizon point.
            cx=(idx.crossline[selected]-xmin)//sx;cy=(idx.inline[selected]-ymin)//sy
            cells=cy*len(x)+cx
            distance=((idx.crossline[selected]-xmin)%sx/sx)**2+((idx.inline[selected]-ymin)%sy/sy)**2
            order=np.lexsort((selected,distance,cells));_,first=np.unique(cells[order],return_index=True)
            selected=selected[order[first]]
        for row in selected:
            position = float(np.clip(sample[row], 0, idx.ns-1))
            if method == "nearest": value = self.values(row, int(np.floor(position+.5)))[0]
            else:
                exact = int(round(position))
                if abs(position-exact) <= 1e-7: value = self.values(row, exact)[0]
                else:
                    low = int(np.floor(position)); pair = self.values(row, slice(low, low+2))
                    value = float(pair[0] + (pair[1]-pair[0]) * (position-low)) if np.isfinite(pair).all() else np.nan
            values[(idx.inline[row]-ymin)//sy, (idx.crossline[row]-xmin)//sx] = value
            self.last_point_types[(idx.inline[row]-ymin)//sy, (idx.crossline[row]-xmin)//sx]=kinds[row]
        self.last_request.update(x_stride=sx, y_stride=sy, output_cells=int(values.size), horizon_points=int(covered.sum()),horizon_in_range_points=int(in_range.sum()),valid_pressure_points=int(np.isfinite(values).sum()),horizon_display_traces=len(selected))
        self.last_request.update(original_horizon_points=int((kinds==1).sum()),interpolated_horizon_points=int((kinds==2).sum()))
        require_fresh(idx.path)
        return Slice(values, x, y, "Crossline", "Inline", f"Along horizon: {name}", False)
