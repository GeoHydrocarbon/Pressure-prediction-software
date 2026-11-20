"""Depth matching for sparse pressure measurements; no extrapolation or gap filling."""
from copy import deepcopy
import numpy as np
import pandas as pd


MATCH_REASONS = {"matched_exact": "同深度预测点", "matched_interpolated": "相邻有效点线性插值",
    "invalid_measurement": "井名、深度或实测压力无效", "well_not_found": "测井数据中没有该井",
    "outside_range": "超出该井预测深度范围", "gap": "相邻点无效或超过允许间隔",
    "ambiguous_depth": "相同深度存在不同预测值", "invalid_curve_point": "对应深度预测无效"}


def prediction_defaults(project):
    data = project.settings["data"]
    cfg = {"well_file": data.get("well_file",''), "columns": deepcopy(data["columns"]), "units": deepcopy(data["units"]),
           "measurement_file": "",
           "measurement_columns": {"well": "井", "Depth": "Depth", "Pp": "Pp"},
           "measurement_units": {"Depth": "m", "Pp": "MPa"},
           "independence_confirmed": False, "max_gap_m": 0.0}
    for key, value in project.settings.get("well_prediction", {}).items():
        if key not in ("depth_reference", "measurements_depth_confirmed"): cfg[key] = deepcopy(value)
    return cfg


def gap_limit(depth, maximum=0):
    if maximum > 0: return float(maximum)
    differences = np.diff(np.unique(np.asarray(depth)[np.isfinite(depth)]))
    positive = differences[differences > 0]
    return float(5 * np.median(positive)) if len(positive) else 0.0


def match_pressure_points(curve, points, max_gap_m=0):
    if not np.isfinite(max_gap_m) or max_gap_m < 0: raise ValueError("最大匹配间隔必须为非负米数；0 表示自动")
    result = points.copy().reset_index(drop=True)
    result["Pp_pred"] = np.nan; result["match_status"] = "invalid_measurement"
    result["left_depth_m"] = np.nan; result["right_depth_m"] = np.nan
    result["max_gap_m"] = np.nan
    groups = {str(well): group for well, group in curve.groupby("well", sort=False)}
    prepared = {}
    for well, group in groups.items():
        group = group.loc[np.isfinite(group.Depth) & (group.Depth >= 0)].sort_values("Depth", kind="stable")
        z, starts = np.unique(group.Depth.to_numpy(dtype=float), return_index=True)
        values = group.Pp_pred.to_numpy(dtype=float)
        if len(starts):
            minima, maxima = np.minimum.reduceat(values, starts), np.maximum.reduceat(values, starts)
            finite = np.isfinite(minima) & np.isfinite(maxima)
            consistent = finite & ((maxima - minima) <= 1e-6)
            predictions = np.where(consistent, values[starts], np.nan)
            reasons = np.where(finite & ~consistent, "ambiguous_depth", "invalid_curve_point")
        else: predictions, reasons = np.array([]), np.array([])
        prepared[well] = (z, predictions, reasons, gap_limit(group.Depth.to_numpy(), max_gap_m))
    # Vectorize point matching within each well; dense labels remain fast too.
    count = len(result); values = np.full(count, np.nan); left_depth = np.full(count, np.nan)
    right_depth = np.full(count, np.nan); limits = np.full(count, np.nan)
    status = np.full(count, "invalid_measurement", dtype=object)
    valid = (result.well.notna() & result.well.astype(str).str.strip().ne("") &
             np.isfinite(result.Depth) & (result.Depth >= 0) & np.isfinite(result.Pp) & (result.Pp >= 0))
    for well, group in result.loc[valid].groupby("well", sort=False):
        indices = group.index.to_numpy(); depths = group.Depth.to_numpy(dtype=float)
        if str(well) not in prepared: status[indices] = "well_not_found"; continue
        z, prediction, reasons, threshold = prepared[str(well)]; limits[indices] = threshold
        if not len(z): status[indices] = "invalid_curve_point"; continue
        right = np.searchsorted(z, depths); r = np.clip(right, 0, len(z)-1); l = np.clip(right-1, 0, len(z)-1)
        r_exact = np.abs(z[r] - depths) <= 1e-6; l_exact = np.abs(z[l] - depths) <= 1e-6
        exact = r_exact | l_exact; nodes = np.where(r_exact, r, l)
        exact_rows, exact_nodes = indices[exact], nodes[exact]
        left_depth[exact_rows] = z[exact_nodes]; right_depth[exact_rows] = z[exact_nodes]
        good = np.isfinite(prediction[exact_nodes])
        values[exact_rows[good]] = prediction[exact_nodes[good]]
        status[exact_rows[good]] = "matched_exact"; status[exact_rows[~good]] = reasons[exact_nodes[~good]]
        inside = ~exact & (right > 0) & (right < len(z))
        status[indices[~exact & ~inside]] = "outside_range"
        rows, a, b, depths_inside = indices[inside], l[inside], r[inside], depths[inside]
        left_depth[rows] = z[a]; right_depth[rows] = z[b]; status[rows] = "gap"
        good = np.isfinite(prediction[a]) & np.isfinite(prediction[b]) & ((z[b] - z[a]) <= threshold)
        values[rows[good]] = prediction[a[good]] + (prediction[b[good]] - prediction[a[good]]) * (depths_inside[good] - z[a[good]]) / (z[b[good]] - z[a[good]])
        status[rows[good]] = "matched_interpolated"
    result["Pp_pred"] = values; result["match_status"] = status
    result["left_depth_m"] = left_depth; result["right_depth_m"] = right_depth; result["max_gap_m"] = limits
    result["supervised_valid"] = result.match_status.isin(["matched_exact", "matched_interpolated"])
    result["error_MPa"] = result.Pp_pred - result.Pp
    result["abs_error_MPa"] = result.error_MPa.abs()
    result["match_reason"] = result.match_status.map(MATCH_REASONS)
    return result


def well_roles(metadata, names, confirmed=False):
    train = set(metadata.get("preprocessing", {}).get("fit_wells", []))
    train |= set(metadata.get("split", {}).get("train", {}).get("wells", []))
    development = set(metadata.get("project", {}).get("data", {}).get("development_wells", []))
    development |= set(metadata.get("split", {}).get("validation", {}).get("wells", []))
    lineage_known = bool(train or development)
    return {str(well): "training_review" if well in train else "development_review" if well in development else
            "independent_validation_user_confirmed" if confirmed and lineage_known else "independence_unverified" for well in names}


ROLE_LABELS = {"training_review": "训练井复核", "development_review": "开发/选参井复核",
               "independent_validation_user_confirmed": "独立验证（用户确认）", "independence_unverified": "独立性未确认"}
ROLE_ENGLISH = {"training_review": "Training-well review", "development_review": "Development / selection-well review",
                "independent_validation_user_confirmed": "Independent validation (user confirmed)", "independence_unverified": "Independence not confirmed"}
