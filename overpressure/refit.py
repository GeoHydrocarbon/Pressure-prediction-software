from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from .config import FEATURES, Project, selected_features, feature_units
from .data import PreparedData, audit_wells, development_fingerprint, sha256_file
from .training import train


def refit(project: Project, cv_folder, *, smoke=False, cancel_file=None):
    """Refit all development wells for a CV-derived, fixed number of epochs."""
    summary_path = Path(cv_folder).resolve() / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "completed":
        raise ValueError("留一井验证尚未完整完成，不能据此最终重训")
    if summary.get("purpose") not in ("cross_validation", "smoke_test"):
        raise ValueError("所选目录不是留一井验证成果")
    if not smoke and summary["purpose"] == "smoke_test":
        raise ValueError("小样本验证记录不能用于正式最终重训，请先完成正式留一井验证")
    cfg = project.settings; previous = summary["project"]
    features=selected_features(cfg)
    if features!=selected_features(previous) or cfg["data"].get("target_curve","Pp")!=previous["data"].get("target_curve","Pp"):raise ValueError('训练曲线或目标与验证记录不同，请重新验证')
    for key in ("algorithm", "training", "physics"):
        if cfg[key] != previous[key]:
            raise ValueError(f"当前 {key} 设置与验证记录不同，请重新进行留一井验证")
    for key in ("columns", "units"):
        if cfg["data"][key] != previous["data"][key]:
            raise ValueError(f"当前数据 {key} 与验证记录不同，请重新验证")
    for key in ("development_wells", "blind_wells"):
        if set(cfg["data"][key]) != set(previous["data"][key]):
            raise ValueError("当前开发井或盲井划分与验证记录不同")
    wells = cfg["data"]["development_wells"]
    folds = summary["folds"]
    if len(folds) != len(wells) or {row["validation_well"] for row in folds} != set(wells):
        raise ValueError("验证记录没有恰好覆盖每口开发井一次")
    epoch_values = [row["best_epoch"] for row in folds]
    if any(isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= cfg["training"]["epochs"] for n in epoch_values):
        raise ValueError("各折最佳轮数记录无效")
    audit = audit_wells(project)
    if "development_sha256" in summary:
        if development_fingerprint(audit.frame, wells,features) != summary["development_sha256"]:
            raise ValueError("开发井数据已经变化，请重新验证")
    elif audit.report["source_sha256"] != summary["source_sha256"]:
        raise ValueError("数据文件与旧版验证记录不一致，请重新验证")
    frame = audit.frame.loc[audit.frame.well.isin(wells) & audit.frame.supervised_valid].copy()
    if set(frame.well) != set(wells):
        raise ValueError("至少一口开发井没有有效训练样本")
    frame["role"] = "train"
    prepared = PreparedData(audit, {"train": frame}, StandardScaler().fit(frame[list(features)].to_numpy()))
    selection = {"cv_summary": str(summary_path), "cv_summary_sha256": sha256_file(summary_path),
                 "epochs": int(np.median(epoch_values)), "rule": "floor(median(best_epoch))",
                 "development_sha256": development_fingerprint(audit.frame, wells,features),
                 "cv_purpose": summary["purpose"], "cv_pooled_metrics": summary.get("pooled_metrics"),
                 "all_development_wells": list(wells)}
    return train(Project(project.path, deepcopy(cfg)), smoke=smoke, cancel_file=cancel_file, _prepared=prepared, _refit_selection=selection)
