"""Paired-point regression evaluation shared by training, charts and exports."""
import numpy as np
import pandas as pd


MAPE_RULE = "MAPE (%) = mean(abs(prediction-reference)/abs(reference))*100；仅排除真实值为零的配对点，不用极小数替代分母。RMSE/MAE/R²/Bias 仍保留真实值为零的有效点。"


def metrics(target, prediction):
    y, p = np.asarray(target, dtype=float), np.asarray(prediction, dtype=float)
    if y.shape != p.shape:
        raise ValueError("真实值与预测值的形状必须一致")
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    result = dict(count=len(y), mae=None, rmse=None, bias=None, r2=None,
                  mape=None, mape_count=0, mape_excluded_zero_count=0)
    if not len(y):
        return result
    error = p - y
    denominator = np.sum((y - y.mean()) ** 2)
    nonzero = y != 0
    result.update(mae=float(np.abs(error).mean()), rmse=float(np.sqrt((error ** 2).mean())),
                  bias=float(error.mean()), r2=float(1 - np.sum(error ** 2) / denominator) if denominator > 0 else None,
                  mape=float((np.abs(error[nonzero] / y[nonzero])).mean() * 100) if nonzero.any() else None,
                  mape_count=int(nonzero.sum()), mape_excluded_zero_count=int((~nonzero).sum()))
    return result


def paired_points(frame):
    """Use exactly the same reference pairs for a scatter and its metrics."""
    if not {"Pp", "Pp_pred"} <= set(frame.columns):
        return frame.iloc[:0].copy()
    frame = frame.copy()
    for key in ("Pp", "Pp_pred"):
        frame[key] = pd.to_numeric(frame[key], errors="coerce")
    valid = np.isfinite(frame.Pp) & np.isfinite(frame.Pp_pred) & (frame.Pp >= 0)
    if "supervised_valid" in frame:
        valid &= frame.supervised_valid.astype(str).str.lower().isin(["true", "1"])
    return frame.loc[valid].copy()


def evaluation_table(frame):
    pairs = paired_points(frame)
    rows = [{"scope": "overall", "well": "全部井（按点汇总）", **metrics(pairs.Pp, pairs.Pp_pred)}] if {"Pp", "Pp_pred"} <= set(frame.columns) else []
    if "well" in frame:
        for well, group in frame.groupby("well", sort=False, dropna=False):
            pairs = paired_points(group)
            value = metrics(pairs.Pp, pairs.Pp_pred) if {"Pp", "Pp_pred"} <= set(pairs.columns) else metrics([], [])
            rows.append({"scope": "well", "well": str(well), **value})
    return pd.DataFrame(rows, columns=["scope", "well", "count", "rmse", "mae", "mape", "r2", "bias", "mape_count", "mape_excluded_zero_count"])
