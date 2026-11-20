"""Map time-domain horizon records to trace geometry without spatial gap filling."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from uuid import uuid4
import numpy as np
import pandas as pd
from .data import read_table, sha256_file
from .jobs import save_json
from .horizon_io import read_horizon_table,MAX_HORIZON_RECORDS


MAX_HORIZON_VALUES = 8_000_000


def geometry_hash(index):
    digest = hashlib.sha256()
    for array in (index.inline, index.crossline): digest.update(np.asarray(array, dtype="<i8").tobytes())
    if index.x is not None:
        for array in (index.x, index.y): digest.update(np.asarray(array, dtype="<f8").tobytes())
    if index.coordinate_units is not None: digest.update(np.asarray(index.coordinate_units, dtype="<i8").tobytes())
    digest.update(index.time_convention.encode())
    return digest.hexdigest()


def map_horizon(raw, index, config):
    mode = config["coordinates"]; columns = config["columns"]
    fields = ("inline", "crossline", "time") if mode == "lines" else ("x", "y", "time") if mode == "xy" else ()
    if not fields: raise ValueError("层位坐标方式必须为 lines 或 xy")
    if len({columns[field] for field in fields}) != len(fields): raise ValueError("层位字段映射不能重复")
    missing = {columns[field] for field in fields} - set(raw.columns)
    if missing: raise ValueError(f"层位文件缺少列：{sorted(missing)}")
    if raw.empty: raise ValueError("层位文件没有数据行")
    if len(raw) > MAX_HORIZON_RECORDS: raise ValueError("层位文件超过 300 万行，请拆分后导入")
    if config["time_unit"] not in ("s", "ms"): raise ValueError("层位时间单位必须为 s 或 ms")
    if config["time_convention"] != index.time_convention: raise ValueError("层位与地震的单/双程时间类型不同；本版不自动乘二或除二")
    if columns.get("name"):
        if columns["name"] not in raw: raise ValueError("找不到层位名称列")
        names = raw[columns["name"]].astype("string").fillna("未命名层位").str.strip().replace("", "未命名层位").astype(str)
    else: names = pd.Series(config.get('default_name','Horizon 1'), index=raw.index)
    layers = names.drop_duplicates().tolist(); n = len(index.inline)
    if len(layers) > 64 or len(layers) * n > MAX_HORIZON_VALUES: raise ValueError("层位数量/映射规模超过上限，请拆分文件（最多 64 层、800 万映射值）")
    source_rows=raw.attrs.get('source_rows',np.arange(len(raw))+2)
    table = pd.DataFrame({"source_row": source_rows, "horizon": names.to_numpy()})
    for key in fields: table[key] = pd.to_numeric(raw[columns[key]], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    table["time_ms"] = table["time"] * (1000 if config["time_unit"] == "s" else 1)
    good = np.isfinite(table[list(fields)].to_numpy()).all(axis=1)
    rows = np.full(len(raw), -1, dtype=np.int64); status = np.full(len(raw), "invalid_value", dtype=object)
    distance = np.full(len(raw), np.nan)
    if mode == "lines":
        good &= (table.inline == np.floor(table.inline)) & (table.crossline == np.floor(table.crossline))
        keys = pd.MultiIndex.from_arrays([index.inline, index.crossline])
        candidate = np.flatnonzero(good)
        rows[candidate] = keys.get_indexer(pd.MultiIndex.from_arrays([table.inline.iloc[candidate], table.crossline.iloc[candidate]]))
        status[candidate] = np.where(rows[candidate] >= 0, "matched", "trace_not_found")
    else:
        if index.x is None or index.y is None: raise ValueError("旧索引不含 XY 坐标，请重新点击“载入切片”建立新版索引")
        if index.coordinate_units is not None and not np.isin(index.coordinate_units, [0, 1]).all():
            raise ValueError("当前道头为角度/经纬度坐标；首版 XY 匹配仅支持同坐标系的平面长度坐标，请改用 Inline/Xline")
        limit = float(config["max_distance"])
        if not np.isfinite(limit) or limit < 0: raise ValueError("XY 匹配距离上限必须为非负数")
        from scipy.spatial import cKDTree
        xy = np.column_stack((index.x, index.y))
        if not np.isfinite(xy).all(): raise ValueError("地震 XY 坐标包含无效数值")
        candidate = np.flatnonzero(good); coordinates = table.loc[candidate, ["x", "y"]].to_numpy()
        distances, nearest = cKDTree(xy).query(coordinates, k=2)
        distance[candidate] = distances[:, 0]
        within = distances[:, 0] <= limit
        ambiguous = np.abs(distances[:, 1]-distances[:, 0]) <= np.maximum(1, distances[:, 0])*1e-10
        matched = within & ~ambiguous
        rows[candidate[matched]] = nearest[matched, 0]
        status[candidate] = np.where(~within, "beyond_distance", np.where(ambiguous, "ambiguous_xy", "matched"))
    times = np.full((len(layers), n), np.nan)
    layer_ids = pd.Categorical(names, categories=layers).codes
    accepted = np.flatnonzero(rows >= 0)
    if len(accepted):
        encoded = layer_ids[accepted].astype(np.int64)*n + rows[accepted]
        order = np.argsort(encoded, kind="stable"); encoded, accepted = encoded[order], accepted[order]
        unique, starts, counts = np.unique(encoded, return_index=True, return_counts=True)
        values = table.time_ms.to_numpy()[accepted]
        conflict = np.maximum.reduceat(values, starts) - np.minimum.reduceat(values, starts) > 1e-6
        group = np.repeat(np.arange(len(starts)), counts)
        status[accepted[conflict[group]]] = "conflicting_time"
        status[accepted[~conflict[group] & (counts[group] > 1)]] = "duplicate_same_time"
        times.reshape(-1)[unique[~conflict]] = values[starts[~conflict]]
    table["trace_index"] = rows; table["match_status"] = status; table["xy_distance"] = distance
    meanings = {"matched": "已匹配", "invalid_value": "坐标或时间无效/线号非整数", "trace_not_found": "地震中无该线号交点",
                "beyond_distance": "超过 XY 距离上限", "ambiguous_xy": "与多条道等距，无法唯一定位",
                "conflicting_time": "同层位同一道存在冲突时间", "duplicate_same_time": "相同时间重复点，合并使用"}
    table["match_reason"] = table.match_status.map(meanings)
    local = (table.time_ms.to_numpy()-index.t0_ms)/index.interval_ms - index.offsets[np.maximum(rows, 0)]
    table["outside_time_range"] = (rows >= 0) & np.isfinite(local) & ((local < -1e-7) | (local > index.ns-1+1e-7))
    table.loc[table.outside_time_range & table.match_status.isin(["matched", "duplicate_same_time"]), "match_reason"] = "已定位，但时间超出该道范围（不取压力）"
    return layers, times, table


@dataclass
class HorizonMap:
    names: list
    times_ms: np.ndarray
    geometry: str
    point_types: np.ndarray | None = None

    @classmethod
    def load(cls, path, index):
        with np.load(path, allow_pickle=False) as data:
            result = cls(data["names"].tolist(), data["times_ms"], str(data["geometry"].item()),data['point_types'] if 'point_types' in data else None)
        if result.geometry != geometry_hash(index) or result.times_ms.shape != (len(result.names), len(index.inline)):
            raise ValueError("层位映射不属于当前地震几何，请重新导入")
        if result.point_types is None:result.point_types=np.where(np.isfinite(result.times_ms),1,0).astype(np.uint8)
        if result.point_types.shape!=result.times_ms.shape or not np.isin(result.point_types,[0,1,2]).all() or not np.array_equal(result.point_types>0,np.isfinite(result.times_ms)):raise ValueError('层位原始点/插值点标记无效')
        return result

    def layer(self, name):
        if name not in self.names: raise ValueError("所选层位不存在")
        return self.times_ms[self.names.index(name)]


def import_horizons(index, config):
    config=dict(config)
    source = Path(config["path"]).resolve();raw=read_horizon_table(source,config.get('text_options'));layers, times, table = map_horizon(raw, index, config)
    if not config.get('point_type_column') and 'point_type' in raw and raw['point_type'].isin(['original','interpolated']).all():
        config['point_type_column']='point_type'
        report_path=source.with_suffix('.json')
        if report_path.is_file():
            try:
                previous=json.loads(report_path.read_text(encoding='utf-8'))
                if previous.get('method')=='linear_delaunay':config['interpolation_report']=str(report_path)
            except (ValueError,OSError):pass
    point_types=np.where(np.isfinite(times),1,0).astype(np.uint8)
    if config.get('point_type_column'):
        column=config['point_type_column']
        if column not in raw:raise ValueError('派生层位缺少原始点/插值点标记列')
        kinds=raw[column].astype(str).str.strip().map({'original':1,'interpolated':2})
        if kinds.isna().any():raise ValueError('层位点标记必须为 original 或 interpolated')
        point_types=np.where(np.isfinite(times),3,0).astype(np.uint8)
        layer_ids=pd.Categorical(table.horizon,categories=layers).codes.astype(np.int64)
        accepted=table.match_status.isin(['matched','duplicate_same_time']).to_numpy()
        cells=layer_ids[accepted]*len(index.inline)+table.trace_index.to_numpy()[accepted]
        np.minimum.at(point_types.reshape(-1),cells,kinds.to_numpy(dtype=np.uint8)[accepted]);point_types[point_types==3]=1
    folder = index.path.parent / "horizons"; folder.mkdir(exist_ok=True)
    target = folder / (uuid4().hex + ".npz")
    np.savez_compressed(target, names=np.array(layers), times_ms=times, geometry=np.array(geometry_hash(index)),point_types=point_types)
    table.to_csv(target.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    mapped = np.isfinite(times)
    local = (times-index.t0_ms)/index.interval_ms - index.offsets[None, :]
    summary = {"path": str(target), "names": layers, "source": str(source), "source_sha256": sha256_file(source), "settings": config,
               "records": len(table), "status_counts": table.match_status.value_counts().to_dict(),
               "mapped_trace_values": int(mapped.sum()), "outside_time_values": int((mapped & ((local < -1e-7) | (local > index.ns-1+1e-7))).sum()),
               "spatial_interpolation": bool(config.get('point_type_column')), "xy_units": "same as scaled seismic coordinates; no CRS conversion",
               'derived':bool(config.get('point_type_column')),'original_points':int((point_types==1).sum()),'interpolated_points':int((point_types==2).sum()),'interpolation_report':config.get('interpolation_report','')}
    save_json(target.with_suffix(".json"), summary)
    return summary


def overlay_line(index, times, section, mode, fixed_value):
    variable = index.crossline if mode == "inline" else index.inline
    fixed = index.inline if mode == "inline" else index.crossline
    selected = np.flatnonzero(fixed == int(fixed_value))
    result = np.full(len(section.x), np.nan)
    positions = np.searchsorted(section.x, variable[selected])
    keep = positions < len(section.x); selected, positions = selected[keep], positions[keep]
    keep = section.x[positions] == variable[selected]; selected, positions = selected[keep], positions[keep]
    local = (times[selected]-index.t0_ms)/index.interval_ms - index.offsets[selected]
    keep = np.isfinite(local) & (local >= -1e-7) & (local <= index.ns-1+1e-7)
    result[positions[keep]] = times[selected[keep]]
    return result
