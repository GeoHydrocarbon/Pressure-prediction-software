"""Well positions, depth/time alignment and bounded sampling along a well path."""
from pathlib import Path
from dataclasses import dataclass
from uuid import uuid4
import json
import numpy as np
import pandas as pd
from .config import Project, unit_factor
from .data import read_table, sha256_file
from .time_depth import TimeDepth
from .horizons import geometry_hash
from .jobs import save_json


MAX_WELL_VALUES = 8_000_000


def read_asset(asset):
    columns = asset["columns"]
    raw = read_table(Path(asset["path"]), dtype={columns["well"]: "string"} if columns.get('well') else None)
    mapped=[v for v in columns.values() if v]
    if len(set(mapped)) != len(mapped): raise ValueError('井名、深度和 XY 字段不能对应同一列')
    missing = set(mapped)-set(raw.columns)
    if missing: raise ValueError(f"{Path(asset['path']).name} 缺少列：{sorted(missing)}")
    rows=np.arange(len(raw))+2
    if asset.get('well_name') and columns.get('well'):
        mask=raw[columns['well']].astype('string').str.strip().eq(asset['well_name']).fillna(False).to_numpy();raw=raw.loc[mask];rows=rows[mask]
    if asset.get('excluded_wells') and columns.get('well'):
        mask=~raw[columns['well']].astype('string').str.strip().isin(asset['excluded_wells']).to_numpy();raw=raw.loc[mask];rows=rows[mask]
    names=raw[columns['well']].astype('string').str.strip() if columns.get('well') else pd.Series(asset.get('well_name',''),index=raw.index,dtype='string')
    frame = pd.DataFrame({'well':names.to_numpy(), 'source_row':rows})
    raw=raw.reset_index(drop=True)
    frame['well'] = frame.well.replace('',pd.NA)
    fields = {"coordinates": ("x", "y"), "trajectory": ("depth", "x", "y"), "curves": ("depth",)}[asset["kind"]]
    for field in fields:
        if columns[field] not in raw: raise ValueError(f"{Path(asset['path']).name} 缺少 {field} 列")
        values = pd.to_numeric(raw[columns[field]], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        frame[field] = np.where(np.isfinite(values),values,np.nan)
    if "depth" in frame: frame["depth"] *= unit_factor("Depth", asset.get("depth_unit", "m"))
    return raw, frame


def interpolate_path(stations, depths, max_gap=0):
    stations = stations.loc[np.isfinite(stations.depth)].sort_values("depth", kind="stable")
    z = stations.depth.unique(); coords = []
    for _, rows in stations.groupby("depth", sort=True):
        xy = rows[["x", "y"]].to_numpy(dtype=float)
        coords.append(xy[0] if np.isfinite(xy).all() and np.allclose(xy, xy[0], atol=1e-6, rtol=0) else [np.nan, np.nan])
    xy = np.array(coords).reshape(-1, 2); result = np.full((len(depths), 2), np.nan)
    if not len(z): return result
    r = np.searchsorted(z, depths); a, b = np.clip(r-1, 0, len(z)-1), np.clip(r, 0, len(z)-1)
    exact = np.abs(z[b]-depths) <= 1e-6
    result[exact] = xy[b[exact]]
    left_exact = ~exact & (np.abs(z[a]-depths) <= 1e-6); result[left_exact] = xy[a[left_exact]]
    gap = float(max_gap) if max_gap > 0 else (5 * float(np.median(np.diff(z))) if len(z) > 1 else 0)
    inside = ~exact & ~left_exact & (r > 0) & (r < len(z)) & ((z[b]-z[a]) <= gap)
    good = inside & np.isfinite(xy[a]).all(axis=1) & np.isfinite(xy[b]).all(axis=1)
    weight = (depths[good]-z[a[good]]) / (z[b[good]]-z[a[good]])
    result[good] = xy[a[good]] + weight[:, None]*(xy[b[good]]-xy[a[good]])
    return result


def sample_well_pressure(reader, rows, times_ms):
    """Coalesce the required samples per trace; keep memory bounded to a trace chunk."""
    idx = reader.index; rows = np.asarray(rows); times_ms = np.asarray(times_ms)
    result = np.full(len(rows), np.nan)
    position = (times_ms-idx.t0_ms)/idx.interval_ms - idx.offsets[np.maximum(rows, 0)]
    valid = (rows >= 0) & np.isfinite(position) & (position >= -1e-7) & (position <= idx.ns-1+1e-7)
    reader.last_request = {"sample_values_read": 0, "traces_read": 0}
    for row in np.unique(rows[valid]):
        selected = np.flatnonzero(valid & (rows == row)); p = np.clip(position[selected], 0, idx.ns-1)
        low = np.floor(p).astype(int); high = np.ceil(p).astype(int)
        exact = np.abs(p-np.round(p)) <= 1e-7
        low[exact] = high[exact] = np.round(p[exact]).astype(int)
        required = np.unique(np.r_[low, high]); values = np.full(len(required), np.nan)
        # Contiguous sample runs avoid repeated scalar reads without loading a volume.
        breaks = np.r_[0, np.flatnonzero(np.diff(required) != 1)+1, len(required)]
        for start, stop in zip(breaks[:-1], breaks[1:]):
            for block in range(start, stop, 65536):
                end = min(block+65536, stop)
                values[block:end] = reader.values(row, slice(int(required[block]), int(required[end-1])+1))
        v0, v1 = values[np.searchsorted(required, low)], values[np.searchsorted(required, high)]
        result[selected] = np.where(np.isfinite(v0) & np.isfinite(v1), v0+(v1-v0)*(p-low), np.nan)
    return result


@dataclass
class WellMap:
    wells: np.ndarray
    depth: np.ndarray
    time_ms: np.ndarray
    x: np.ndarray
    y: np.ndarray
    trace_row: np.ndarray
    curve_names: list
    curves: np.ndarray
    pressure_names: np.ndarray
    well_pressure: np.ndarray
    seismic_pressure: np.ndarray
    geometry: str
    path: Path | None = None

    @classmethod
    def load(cls, path, index):
        with np.load(path, allow_pickle=False) as data:
            result = cls(data["wells"], data["depth"], data["time_ms"], data["x"], data["y"], data["trace_row"], data["curve_names"].tolist(),
                         data["curves"], data["pressure_names"], data["well_pressure"], data["seismic_pressure"], str(data["geometry"].item()), Path(path))
        if result.geometry != geometry_hash(index): raise ValueError("井映射不属于当前地震几何，请重新加载井数据")
        if result.curves.shape != (len(result.curve_names), len(result.wells)): raise ValueError("井曲线缓存尺寸错误")
        return result

    def table(self, well=None):
        selected = np.ones(len(self.wells), dtype=bool) if well is None else self.wells == well
        return pd.DataFrame({"well": self.wells[selected], "depth_m": self.depth[selected], "time_ms": self.time_ms[selected],
            "X": self.x[selected], "Y": self.y[selected], "trace_index": self.trace_row[selected],
            "well_pressure_MPa": self.well_pressure[selected], "seismic_pressure_MPa": self.seismic_pressure[selected],
            "seismic_minus_well_MPa": self.seismic_pressure[selected]-self.well_pressure[selected]})


def prepare_wells(reader, config):
    from scipy.spatial import cKDTree
    idx = reader.index
    if idx.x is None: raise ValueError("请重新载入切片，建立含 XY 的新版地震索引")
    if idx.coordinate_units is not None and not np.isin(idx.coordinate_units, [0, 1]).all(): raise ValueError("井定位首版支持与地震一致的平面长度坐标，不自动转换经纬度")
    assets = config["assets"]; frames, heads, trajectories, provenance = [], [], [], []
    for number, asset in enumerate(assets):
        raw, frame = read_asset(asset); provenance.append({"kind": asset["kind"], "path": asset["path"], "sha256": sha256_file(Path(asset["path"]))})
        if asset["kind"] == "coordinates": heads.append(frame); continue
        if asset["kind"] == "trajectory": frame["relative"] = asset.get("relative", False); trajectories.append(frame); continue
        frame["source_file"] = asset["path"]; frame["asset_id"] = number
        time_column = asset.get("time_column", "")
        if time_column:
            if asset.get("time_convention", "twt") != idx.time_convention: raise ValueError("井曲线时间列与地震的单/双程时间类型不同")
            frame["time_ms"] = pd.to_numeric(raw[time_column], errors="coerce").to_numpy(dtype=float, na_value=np.nan) * (1000 if asset.get("time_unit") == "s" else 1)
        else: frame["time_ms"] = np.nan
        frame["direct_time"] = bool(time_column)
        pressure_column = asset.get("pressure_column", ""); frame["pressure_name"] = asset.get('curve_name',pressure_column) if pressure_column else ''
        frame["well_pressure"] = pd.to_numeric(raw[pressure_column], errors="coerce").to_numpy(dtype=float, na_value=np.nan) * unit_factor("Pp", asset.get("pressure_unit", "MPa")) if pressure_column else np.nan
        frame.loc[~np.isfinite(frame.well_pressure), 'well_pressure'] = np.nan
        for column in raw.columns:
            if asset.get('curve_column') and column!=asset['curve_column']:continue
            if str(column) not in (asset["columns"]["well"], asset["columns"]["depth"], time_column):
                values = pd.to_numeric(raw[column], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
                values = np.where(np.isfinite(values),values,np.nan)
                if np.isfinite(values).any():
                    name=asset.get('curve_name',str(column))
                    if name in ('AC','DEN','Ip'):values=values*unit_factor(name,asset['curve_unit']) if 'curve_unit' in asset else values
                    frame['curve:'+name] = values
        frames.append(frame)
    if not frames: raise ValueError("请在项目井数据中先添加测井曲线或井预测结果")
    logs = pd.concat(frames, ignore_index=True).sort_values(["well", "asset_id", "depth"], kind="stable").reset_index(drop=True)
    curve_columns = [column for column in logs if column.startswith("curve:")]
    if len(logs) > 500000 or len(curve_columns)*len(logs) > MAX_WELL_VALUES: raise ValueError("井曲线规模过大，请减少文件/曲线（最多 50 万点、800 万曲线值）")
    if not curve_columns: raise ValueError("没有可显示的数值曲线")
    indirect = ~logs.direct_time
    if indirect.any():
        bounds=(idx.t0_ms/1000,(idx.t0_ms+(idx.nt-1)*idx.interval_ms)/1000)
        report_path=idx.path.parent/'result.json'
        if report_path.is_file():
            predicted_bounds=json.loads(report_path.read_text(encoding='utf-8')).get('predicted_time_range_seconds')
            if predicted_bounds is not None:bounds=predicted_bounds
        td = TimeDepth(Project(Path(config["project_path"]), {"time_depth": config["time_depth"]}),time_bounds_seconds=bounds)
        if td.config["time_convention"] != idx.time_convention: raise ValueError("时深关系与地震的单/双程时间类型不同")
        logs.loc[indirect, "time_ms"] = td.inverse(logs.loc[indirect, "depth"].to_numpy()) * 1000
    positions = np.full((len(logs), 2), np.nan); modes = {}
    head_table = pd.concat(heads, ignore_index=True) if heads else pd.DataFrame(columns=["well", "x", "y"])
    trajectory_table = pd.concat(trajectories, ignore_index=True) if trajectories else pd.DataFrame(columns=["well", "depth", "x", "y", "relative"])
    for well, group in logs.groupby("well", sort=False):
        h = head_table.loc[head_table.well == well, ["x", "y"]].to_numpy(dtype=float)
        head = h[0] if len(h) and np.isfinite(h).all() and np.allclose(h, h[0], atol=1e-6, rtol=0) else None
        stations = trajectory_table.loc[trajectory_table.well == well].copy()
        if len(stations):
            if stations.relative.any():
                if head is None: modes[str(well)] = "相对轨迹缺少唯一井口坐标"; continue
                stations.loc[stations.relative, ["x", "y"]] += head
            positions[group.index] = interpolate_path(stations, group.depth.to_numpy(), config.get("trajectory_max_gap_m", 0))
            modes[str(well)] = "按井轨迹定位，覆盖外不外推"
        elif head is not None and config.get("vertical_fallback", True): positions[group.index] = head; modes[str(well)] = "无轨迹，按井口垂直井定位"
        else: modes[str(well)] = "缺少可用井坐标或轨迹"
    distance_limit = float(config.get("max_distance", 25))
    if not np.isfinite(distance_limit) or distance_limit < 0: raise ValueError("井到地震道最大距离必须为非负数")
    rows = np.full(len(logs), -1, dtype=np.int64); distances = np.full(len(logs), np.nan)
    valid = np.isfinite(positions).all(axis=1) & logs.well.notna().to_numpy() & np.isfinite(logs.depth.to_numpy())
    candidates = np.flatnonzero(valid)
    if len(candidates):
        d, r = cKDTree(np.column_stack((idx.x, idx.y))).query(positions[candidates], k=2)
        distances[candidates] = d[:, 0]
        unique = np.abs(d[:, 1]-d[:, 0]) > np.maximum(1, d[:, 0])*1e-10
        good = (d[:, 0] <= distance_limit) & unique
        rows[candidates[good]] = r[good, 0]
    if not (rows >= 0).any(): raise ValueError("没有井点可定位到地震道：请检查井名关联、井坐标/轨迹、坐标单位和匹配距离")
    seismic_pressure = sample_well_pressure(reader, rows, logs.time_ms.to_numpy())
    mapping = WellMap(logs.well.fillna("").to_numpy(dtype=str), logs.depth.to_numpy(), logs.time_ms.to_numpy(), positions[:, 0], positions[:, 1], rows,
                      [name[6:] for name in curve_columns], logs[curve_columns].to_numpy(dtype=float).T,
                      logs.pressure_name.fillna("").to_numpy(dtype=str), logs.well_pressure.to_numpy(), seismic_pressure, geometry_hash(idx))
    folder = idx.path.parent / "wells"; folder.mkdir(exist_ok=True); path = folder / (uuid4().hex+".npz")
    np.savez_compressed(path, wells=mapping.wells, depth=mapping.depth, time_ms=mapping.time_ms, x=mapping.x, y=mapping.y,
                       trace_row=rows, curve_names=np.array(mapping.curve_names), curves=mapping.curves, pressure_names=mapping.pressure_names,
                       well_pressure=mapping.well_pressure, seismic_pressure=seismic_pressure, geometry=np.array(mapping.geometry))
    table = mapping.table(); table["source_file"] = logs.source_file.to_numpy(); table["source_row"] = logs.source_row.to_numpy()
    table["xy_distance"] = distances
    table["trace_inline"] = np.where(rows >= 0, idx.inline[np.maximum(rows, 0)], np.nan)
    table["trace_crossline"] = np.where(rows >= 0, idx.crossline[np.maximum(rows, 0)], np.nan)
    table["status"] = np.where(rows < 0, "无法空间定位", np.where(~np.isfinite(mapping.time_ms), "时间缺失/深度超时深范围", np.where(~np.isfinite(seismic_pressure), "地震无效区或超时间范围", "已配对")))
    table.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    summary = {"path": str(path), "names": list(dict.fromkeys(mapping.wells)), "curves": mapping.curve_names, "rows": len(logs),
               "located_rows": int((rows >= 0).sum()), "pressure_pairs": int((np.isfinite(seismic_pressure) & np.isfinite(mapping.well_pressure)).sum()),
               "well_modes": modes, "sources": provenance, "settings": config, "read_stats": reader.last_request,
               "status_counts": table.status.value_counts().to_dict(), "scope": "井曲线与地震预测的差异，不自动等同于实测精度"}
    save_json(path.with_suffix(".json"), summary)
    return summary


def well_overlay(mapping, index, mode, line, well, curve):
    if mode not in ("inline", "crossline") or curve not in mapping.curve_names: return {}
    selected = mapping.wells == well; rows = mapping.trace_row; fixed = index.inline if mode == "inline" else index.crossline
    variable = index.crossline if mode == "inline" else index.inline
    visible = selected & (rows >= 0) & (fixed[np.maximum(rows, 0)] == int(line)) & np.isfinite(mapping.time_ms)
    chosen = np.flatnonzero(selected)
    center = np.where(visible[chosen], variable[np.maximum(rows[chosen], 0)], np.nan).astype(float)
    values = mapping.curves[mapping.curve_names.index(curve), chosen].copy()
    pressure = mapping.pressure_names[chosen] == curve
    values[pressure] = mapping.well_pressure[chosen][pressure]
    other = np.where(pressure, mapping.seismic_pressure[chosen], np.nan)
    # Keep missing trajectory/curve intervals broken instead of connecting across them.
    depth = mapping.depth[chosen]; delta = np.diff(depth); positive = delta[np.isfinite(delta) & (delta > 0)]
    breaks = np.flatnonzero((delta < 0) | (delta > 5*np.median(positive)))+1 if len(positive) else np.array([], dtype=int)
    return {"center": np.insert(center, breaks, np.nan), "time": np.insert(mapping.time_ms[chosen], breaks, np.nan),
            "curve": np.insert(values, breaks, np.nan), "seismic": np.insert(other, breaks, np.nan)}
