"""Streaming pressure prediction with bounded blocks and full output verification."""
from datetime import datetime
import hashlib
from pathlib import Path
import shutil
import time

import numpy as np

from .config import unit_factor
from .data import canonical_features, sha256_file
from .jobs import check_cancel, checked_sha256, emit, save_json
from .time_depth import TimeDepth


MAX_BLOCK_POINTS = 262144


def update_digests(data_hash, mask_hash, pressure, valid):
    data_hash.update(np.ascontiguousarray(np.where(valid, pressure, 0), dtype="<f4").tobytes())
    mask_hash.update(np.ascontiguousarray(valid, dtype=np.uint8).tobytes())


def predict(project, bundle_path, *, full=False, line_range=None, trace_start=0, trace_count=32, trace_stride=1, block_traces=None, cancel_file=None):
    import segyio
    from .model import ModelBundle
    from .seismic import inspect_seismic

    if block_traces is None: block_traces = project.settings["seismic"].get("block_traces", 32)
    if isinstance(block_traces, bool) or not isinstance(block_traces, int) or not 1 <= block_traces <= 256: raise ValueError("分块道数必须为 1–256 的整数")
    if full and line_range is not None: raise ValueError('全体预测与自定义线号范围不能同时使用')
    if line_range is not None:
        if len(line_range) != 4 or any(isinstance(v, bool) or not isinstance(v, int) for v in line_range): raise ValueError('线号范围必须为四个整数')
        il0, il1, xl0, xl1 = line_range
        if il0 > il1 or xl0 > xl1: raise ValueError('Inline / Crossline 起始线号不能大于结束线号')
    if not full and line_range is None and (trace_start < 0 or not 1 <= trace_count <= 512 or trace_stride < 1):
        raise ValueError("小范围预测要求起始道 >= 0，道数 1–512，道步长 >= 1")
    check_cancel(cancel_file)
    bundle = ModelBundle(bundle_path)
    from .config import FEATURES
    missing=set(bundle.features)-set(FEATURES)
    if missing:raise ValueError('模型需要额外曲线，但当前地震未提供对应属性体：'+', '.join(sorted(missing))+'。不能用其它属性替代。')
    feature_indices=[FEATURES.index(name) for name in bundle.features]
    if bundle.metadata["purpose"] == "smoke_test": raise ValueError("软件链路试算模型不能用于地震压力预测，请选择正式训练模型")
    cfg = project.settings; seismic = cfg["seismic"]; td = TimeDepth(project)
    if td.config['time_convention']!=seismic['time_convention']:raise ValueError('输入时间与时深关系的单/双程时间类型不一致')
    emit("stage", message="正在校验输入地震体的逐道几何与采样")
    inspection = inspect_seismic(project, sample_attributes=False)
    check_cancel(cancel_file)
    if not inspection["passed"]: raise ValueError("；".join(inspection["errors"]))
    info = inspection["files"]["density"]; n, ns = info["traces"], info["samples"]
    if ns > MAX_BLOCK_POINTS: raise ValueError("单道采样点数超过当前分块上限，请先分段处理")
    if line_range is not None:
        with segyio.open(str(project.resolve(seismic['density_file'])), 'r', ignore_geometry=True) as source:
            inline = np.asarray(source.attributes(seismic['header_fields']['inline'])[:])
            crossline = np.asarray(source.attributes(seismic['header_fields']['crossline'])[:])
        source_indices = np.flatnonzero((inline >= il0) & (inline <= il1) & (crossline >= xl0) & (crossline <= xl1))
        trace_count = len(source_indices)
        if not trace_count: raise ValueError('自定义 Inline / Crossline 范围内没有地震道，请核对线号与道头设置')
        selection = dict(mode='line_range', inline=[il0, il1], crossline=[xl0, xl1], count=trace_count,
                         actual_inline=[int(inline[source_indices].min()),int(inline[source_indices].max())],
                         actual_crossline=[int(crossline[source_indices].min()),int(crossline[source_indices].max())])
        emit('stage', message=f'自定义范围匹配到 {trace_count} 道（含起止线号）；按输入文件原始道序预测，缺失交点不补道')
    else:
        if full: trace_start, trace_stride, trace_count = 0, 1, n
        if trace_start + (trace_count - 1) * trace_stride >= n: raise ValueError(f"预测范围超出输入总道数 {n}")
        source_indices = range(trace_start, trace_start + trace_count * trace_stride, trace_stride)
        selection = dict(start=trace_start,count=trace_count,stride=trace_stride)
    volume = full or line_range is not None
    block_size = min(block_traces, max(1, MAX_BLOCK_POINTS // ns))
    out = project.resolve(cfg["output_dir"]) / ("seismic_volumes" if volume else "seismic_trials") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True)
    report = {"status": "running", "purpose": "seismic_volume" if volume else "seismic_trial", "model": str(bundle.folder),"features":list(bundle.features),"feature_units":bundle.feature_units,
              "model_weights_sha256": bundle.metadata["weights_sha256"], "input_inspection": inspection,
              "time_depth": cfg["time_depth"], "project": cfg, "output_domain": "input_time_grid", "pressure_unit": "MPa",
              "null_value": "NaN", "valid_mask": "validity.sgy: 1=predicted, 0=invalid", "shape": [trace_count, ns],
              "block_traces": block_size, "peak_block_points": 0, "outside_training_range_points": 0,
              "predicted_points": 0, "invalid_points": 0, "source_trace_selection": selection}
    if line_range is not None:
        np.save(out / 'source_trace_indices.npy', source_indices, allow_pickle=False)
        report['source_trace_indices_file'] = 'source_trace_indices.npy'
        report['source_trace_indices_sha256'] = sha256_file(out / 'source_trace_indices.npy')
    elif not full: report["source_trace_indices"] = list(source_indices)
    if cfg["time_depth"]["method"] == "table": report["time_depth_table_sha256"] = sha256_file(project.resolve(cfg["time_depth"]["table_file"]))
    report.update(time_selection='input_valid_samples',invalid_attribute_points=0,unavailable_depth_points=0,predicted_time_range_seconds=None)
    emit('stage',message='按输入地震的有效采样点自动预测；缺失属性和时深表未覆盖的位置留空，不使用旧的手动时间范围')
    save_json(out / "result.json", report)
    temp_pressure, temp_mask = out / "pressure.partial.sgy", out / "validity.partial.sgy"
    written_data, written_mask = hashlib.sha256(), hashlib.sha256()
    minimum, maximum = float("inf"), -float("inf")
    try:
        with segyio.open(str(project.resolve(seismic["density_file"])), "r", ignore_geometry=True) as den, segyio.open(str(project.resolve(seismic["impedance_file"])), "r", ignore_geometry=True) as ip:
            needed = 2 * (3600 + den.ext_headers * 3200 + trace_count * (240 + ns * 4)) + 64 * 1024 * 1024
            if shutil.disk_usage(out).free < needed: raise ValueError(f"输出磁盘空间不足，至少需要约 {needed / 1024**3:.2f} GB")
            spec = segyio.spec(); spec.format = 5; spec.sorting = den.sorting
            spec.samples = den.samples; spec.tracecount = trace_count; spec.ext_headers = den.ext_headers
            with segyio.create(str(temp_pressure), spec) as dst, segyio.create(str(temp_mask), spec) as mask_dst:
                for output, title in ((dst, "PORE PRESSURE MPa; TIME DOMAIN; INVALID=NaN"), (mask_dst, "PRESSURE VALIDITY: 1=VALID; 0=INVALID; TIME DOMAIN")):
                    for j in range(den.ext_headers + 1): output.text[j] = den.text[j]
                    text = bytearray(den.text[0]); text[:80] = ("C01 " + title).ljust(80).encode("ascii"); output.text[0] = bytes(text)
                    output.bin = den.bin; output.bin[segyio.BinField.Format] = 5
                started = time.monotonic()
                for start in range(0, trace_count, block_size):
                    check_cancel(cancel_file)
                    rows = range(start, min(start + block_size, trace_count))
                    source_rows = [int(source_indices[i]) for i in rows]
                    density = np.stack([den.trace[i].copy() for i in source_rows]) * unit_factor("DEN", seismic["density_unit"])
                    impedance = np.stack([ip.trace[i].copy() for i in source_rows]) * unit_factor("Ip", seismic["impedance_unit"])
                    delay = np.array([den.header[i][109] for i in source_rows])[:, None] / 1000.
                    seconds = delay + np.arange(ns)[None, :] * info["interval_us"] / 1e6
                    attribute_valid=np.isfinite(density)&np.isfinite(impedance)&(density>0)&(impedance>0)
                    depth=td.transform(seconds,input_convention=seismic['time_convention'],valid_mask=attribute_valid)
                    report['invalid_attribute_points']+=int((~attribute_valid).sum())
                    report['unavailable_depth_points']+=int((attribute_valid&(~np.isfinite(depth)|(depth<0))).sum())
                    features = canonical_features(depth,density,impedance)[...,feature_indices]
                    pred = bundle.predict(features.reshape(-1, len(bundle.features))).reshape(len(source_rows), ns)
                    pred[~(attribute_valid & np.isfinite(depth) & (depth>=0))]=np.nan
                    valid = np.isfinite(pred)
                    if valid.any():
                        bounds=[float(seconds[valid].min()),float(seconds[valid].max())]
                        previous=report['predicted_time_range_seconds']
                        report['predicted_time_range_seconds']=bounds if previous is None else [min(previous[0],bounds[0]),max(previous[1],bounds[1])]
                    low = np.asarray(bundle.metadata["preprocessing"]["minimum"]); high = np.asarray(bundle.metadata["preprocessing"]["maximum"])
                    report["outside_training_range_points"] += int((valid & ((features < low) | (features > high)).any(axis=-1)).sum())
                    report["predicted_points"] += int(valid.sum()); report["invalid_points"] += int(valid.size - valid.sum())
                    report["peak_block_points"] = max(report["peak_block_points"], int(valid.size))
                    if valid.any(): minimum, maximum = min(minimum, float(pred[valid].min())), max(maximum, float(pred[valid].max()))
                    for local, (row, source) in enumerate(zip(rows, source_rows)):
                        for output in (dst, mask_dst): output.header[row] = den.header[source]; output.header[row][203] = 0
                        dst.trace[row] = pred[local]; mask_dst.trace[row] = valid[local].astype(np.float32)
                    update_digests(written_data, written_mask, pred, valid)
                    done = min(start + block_size, trace_count); elapsed = max(time.monotonic() - started, .001)
                    emit("seismic_progress", completed=done, total=trace_count, phase="predict", traces_per_second=done / elapsed, eta_seconds=(trace_count - done) * elapsed / done)
        if not report["predicted_points"]: raise ValueError("所选范围没有有效预测点，请核对属性、单位和时深关系")
        emit("stage", message="正在分块重读全部输出，核对数值、有效区与道头")
        read_data, read_mask = hashlib.sha256(), hashlib.sha256()
        with segyio.open(str(temp_pressure), "r", ignore_geometry=True) as check, segyio.open(str(temp_mask), "r", ignore_geometry=True) as mask, segyio.open(str(project.resolve(seismic["density_file"])), "r", ignore_geometry=True) as source:
            if any(f.tracecount != trace_count or len(f.samples) != ns or int(f.format) != 5 for f in (check, mask)):
                raise ValueError("输出 SEG-Y 结构校验失败")
            fields = {*seismic["header_fields"].values(), 109, 115, 117}
            for start in range(0, trace_count, block_size):
                check_cancel(cancel_file)
                rows = range(start, min(start + block_size, trace_count))
                values = np.stack([check.trace[i].copy() for i in rows]); valid = np.stack([mask.trace[i].copy() for i in rows])
                if np.isinf(values).any() or not np.isin(valid, [0, 1]).all() or not np.array_equal(np.isfinite(values), valid.astype(bool)):
                    raise ValueError("输出有效区与压力值校验失败")
                update_digests(read_data, read_mask, values, valid.astype(bool))
                for row in rows:
                    original = int(source_indices[row])
                    if any(check.header[row][k] != source.header[original][k] or mask.header[row][k] != source.header[original][k] for k in fields):
                        raise ValueError("输出 SEG-Y 道头重读校验失败")
                emit("seismic_progress", completed=min(start + block_size, trace_count), total=trace_count, phase="verify")
        if (read_data.digest(), read_mask.digest()) != (written_data.digest(), written_mask.digest()): raise ValueError("输出压力或掩码完整数值摘要校验失败")
        emit("stage", message="正在计算成果文件校验值")
        p_hash, m_hash = checked_sha256(temp_pressure, cancel_file), checked_sha256(temp_mask, cancel_file)
        for entry in inspection["files"].values():
            stat = Path(entry["path"]).stat()
            if stat.st_size != entry["size_bytes"] or stat.st_mtime_ns != entry["mtime_ns"]: raise ValueError("运行期间输入地震文件发生变化，成果不予发布")
        check_cancel(cancel_file)
        temp_pressure.replace(out / "pressure.sgy"); temp_mask.replace(out / "validity.sgy")
        report.update(status="completed", round_trip_verified=True, pressure_sha256=p_hash, validity_sha256=m_hash,
                      pressure_range=[minimum, maximum], values_sha256=read_data.hexdigest(), mask_values_sha256=read_mask.hexdigest())
        report['valid_attribute_time_range_seconds']=(td.observed_bounds*td.seconds_per_unit).tolist() if td.observed_bounds is not None else None
        save_json(out / "result.json", report)
        emit("completed", artifact="seismic_volume" if volume else "seismic_trial", path=str(out), metrics={"predicted_points": report["predicted_points"]})
    except BaseException as exc:
        from .jobs import Cancelled
        report.update(status="cancelled" if isinstance(exc, (Cancelled, KeyboardInterrupt)) else "failed", error=str(exc)); save_json(out / "result.json", report)
        raise
    return out
