from __future__ import annotations

from contextlib import ExitStack
import numpy as np

from .config import Project, unit_factor
from .data import canonical_features, feature_validity
from .time_depth import TimeDepth


def predict_seismic_trial(project: Project, bundle_path, *, trace_start=0, trace_count=32, trace_stride=1, cancel_file=None, block_traces=None):
    from .predict_volume import predict
    return predict(project, bundle_path, trace_start=trace_start, trace_count=trace_count, trace_stride=trace_stride, cancel_file=cancel_file, block_traces=block_traces)


def predict_seismic_volume(project: Project, bundle_path, *, block_traces=None, cancel_file=None):
    from .predict_volume import predict
    return predict(project, bundle_path, full=True, block_traces=block_traces, cancel_file=cancel_file)


def inspect_seismic(project: Project, *, sample_attributes=True) -> dict:
    import segyio

    cfg = project.settings["seismic"]
    paths = {"density": project.resolve(cfg["density_file"]), "impedance": project.resolve(cfg["impedance_file"])}
    fields = dict(cfg["header_fields"])
    fields.update(delay_ms=109, sample_count=115, interval_us=117)
    td = TimeDepth(project)
    reports, errors, warnings = {}, [], []
    with ExitStack() as stack:
        files = {key: stack.enter_context(segyio.open(str(path), "r", ignore_geometry=True)) for key, path in paths.items()}
        headers = {}
        for key, f in files.items():
            h = {name: np.asarray(f.attributes(field)[:]).copy() for name, field in fields.items()}
            headers[key] = h
            intervals = np.unique(h["interval_us"])
            binary_dt = int(f.bin[segyio.BinField.Interval])
            if binary_dt <= 0 or len(intervals) != 1 or int(intervals[0]) != binary_dt:
                errors.append(f"{key}: 道采样间隔不统一或与二进制头不一致")
            if np.any(h["sample_count"] != len(f.samples)):
                errors.append(f"{key}: 道头采样点数与实际点数不一致")
            if f.tracecount < 1 or len(f.samples) < 2:
                raise ValueError(f"{key}: 空地震数据或采样点数不足")
            if len(np.unique(h["delay_ms"])) != 1:
                warnings.append(f"{key}: 各道起始时间不同，预测时必须逐道构建时间轴")
            pairs = np.stack((h["inline"], h["crossline"]), axis=1)
            duplicate_pairs = len(pairs) - len(np.unique(pairs, axis=0))
            if duplicate_pairs:
                errors.append(f"{key}: 指定 Inline/Crossline 映射出现重复位置，请核对道头或处理叠前数据")
            stat = paths[key].stat()
            reports[key] = {"path": str(paths[key]), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                            "traces": f.tracecount, "samples": len(f.samples), "interval_us": binary_dt,
                            "header_ranges": {name: [int(v.min()), int(v.max())] for name, v in h.items()},
                            "duplicate_line_pairs": duplicate_pairs,
                            "sample_format": int(f.format)}
        base, other = files["density"], files["impedance"]
        if base.tracecount != other.tracecount or len(base.samples) != len(other.samples):
            errors.append("密度与波阻抗的道数或采样点数不同")
        if not np.array_equal(base.samples, other.samples):
            errors.append("密度与波阻抗采样时间轴不同")
        mismatch = {name: (int(np.count_nonzero(headers["density"][name] != headers["impedance"][name]))
                           if base.tracecount == other.tracecount else None) for name in fields}
        if any(value != 0 for value in mismatch.values()):
            errors.append("密度与波阻抗的逐道几何或采样道头不一致")
        geometry_passed = not errors
        sampling = {}
        if not errors and sample_attributes:
            indices = np.unique(np.linspace(0, base.tracecount - 1, min(base.tracecount, int(cfg["audit_sample_traces"])), dtype=int))
            den = np.stack([base.trace[int(i)].copy() for i in indices]) * unit_factor("DEN", cfg["density_unit"])
            ip = np.stack([other.trace[int(i)].copy() for i in indices]) * unit_factor("Ip", cfg["impedance_unit"])
            time = headers["density"]["delay_ms"][indices, None] / 1000.0 + np.arange(len(base.samples))[None, :] * reports["density"]["interval_us"] / 1e6
            try:
                attribute_valid=np.isfinite(den)&np.isfinite(ip)&(den>0)&(ip>0)
                depth = td.transform(time, input_convention=cfg["time_convention"], require_confirmed=False,valid_mask=attribute_valid)
                features = canonical_features(depth, den, ip)
                valid = feature_validity(features)
                dz = np.diff(depth, axis=1)
                finite_dz = dz[np.isfinite(dz)]
                sampling = {"sampled_traces": len(indices), "sampled_points": int(valid.size),
                            "prediction_valid_points": int(valid.sum()), "invalid_points": int((~valid).sum()),
                            "depth_step_range_m": [float(finite_dz.min()), float(finite_dz.max())] if len(finite_dz) else None,
                            "features_quantiles_01_50_99": {name: np.quantile(features[..., j][valid], [.01, .5, .99]).tolist()
                                                                for j, name in enumerate(("Depth", "AC", "DEN", "Ip"))} if valid.any() else {}}
                if not valid.any():
                    errors.append("抽样中没有有效特征点，请核对属性单位、无效区及时深范围")
            except ValueError as exc:
                errors.append(str(exc))
    warnings.append("SEG-Y 属性幅值仅作确定性抽样检查；不代表全体数据质量或井震标定已通过。" if sample_attributes else "本次仅预检全体几何和采样；属性与时深有效性在所选道逐点检查。")
    return {"files": reports, "header_mismatches": mismatch, "sample": sampling,
            "geometry_and_sampling_passed": geometry_passed,
            "errors": errors, "warnings": warnings, "passed": not errors, "sample_attributes_checked": sample_attributes}
