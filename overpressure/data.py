from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import FEATURES, Project, unit_factor, selected_features, feature_units


def read_table(path: Path, *, dtype=None, nrows=None) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path, engine="openpyxl", dtype=dtype, nrows=nrows)
    if path.suffix.lower() != ".csv":
        raise ValueError("井数据和时深表支持 .xlsx 或 .csv")
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding, dtype=dtype, nrows=nrows)
        except UnicodeDecodeError:
            pass
    raise ValueError(f"无法解码 CSV: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def development_fingerprint(frame, wells, features=FEATURES):
    selected = frame.loc[frame.well.isin(wells), ["well", "source_row", *features, "Pp"]]
    return hashlib.sha256(selected.to_csv(index=False, float_format="%.17g", na_rep="NaN", lineterminator="\n").encode("utf-8")).hexdigest()


def canonical_features(depth, density, impedance) -> np.ndarray:
    """Physical inputs already in m, g/cc and g/cc*m/s; same AC rule for all paths."""
    depth, density, impedance = np.broadcast_arrays(
        np.asarray(depth, dtype=float), np.asarray(density, dtype=float), np.asarray(impedance, dtype=float)
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ac = 304800.0 * density / impedance
    return np.stack((depth, ac, density, impedance), axis=-1)


def feature_validity(features: np.ndarray, names=FEATURES) -> np.ndarray:
    features = np.asarray(features)
    valid=np.isfinite(features).all(axis=-1)
    for i,name in enumerate(names):
        if name=='Depth':valid &= features[...,i]>=0
        elif name in ('AC','DEN','Ip'):valid &= features[...,i]>0
    return valid


@dataclass
class WellAudit:
    frame: pd.DataFrame
    issues: pd.DataFrame
    report: dict


@dataclass
class PreparedData:
    audit: WellAudit
    splits: dict[str, pd.DataFrame]
    scaler: StandardScaler


def audit_wells(project: Project, *, require_labels=True, require_configured_wells=True) -> WellAudit:
    cfg = project.settings["data"]
    features=selected_features(project.settings)
    library=cfg.get('training_selection')=='well_library'
    if library:
        from .well_library import read_library_frame
        frame,sources,source_digest=read_library_frame(project,cfg['development_wells'],features,cfg.get('target_curve','Pp'),require_labels=require_labels)
        paths=[Path(source['path']) for source in sources]
        mapping={k:k for k in ('well','Depth',*features,'Pp')};raw=frame.copy()
    else:
        from .well_io import selected_paths,read_well_files
        paths=selected_paths(project,cfg);mapping=dict(cfg['columns'])
        for key in features:mapping.setdefault(key,key)
        raw,sources,source_digest=read_well_files(paths,mapping,required=['well','Depth',*features]+(['Pp'] if require_labels else []))
        frame = pd.DataFrame(index=raw.index)
        frame['source_row'] = np.arange(len(raw)) + 2
        frame['source_file']=raw['_source_file'];frame['source_file_row']=raw['_source_file_row']
        frame['well'] = raw[mapping['well']].astype('string').str.strip()
    invalid_names = frame.well.isna() | frame.well.eq("")
    issues = []
    for field in dict.fromkeys(("Depth",*features,"Pp")):
        original = raw[mapping[field]] if mapping[field] in raw else pd.Series(np.nan, index=raw.index)
        numeric = pd.to_numeric(original, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        values = numeric if library else numeric * (unit_factor(field,cfg["units"][field]) if field in ("Depth","AC","DEN","Ip","Pp") else 1.)
        frame[field] = values
        bad = ~np.isfinite(values)
        domain = np.isfinite(values) & ((values < 0) if field in ("Depth", "Pp") else (values <= 0) if field in ("AC","DEN","Ip") else np.zeros(len(values),dtype=bool))
        for idx in np.flatnonzero(bad | domain):
            issues.append({"well": str(frame.well.iloc[idx]), "source_row": int(idx + 2),
                           "field": field, "source_value": str(original.iloc[idx]),
                           "reason": "non_finite_or_not_numeric" if bad[idx] else "outside_physical_domain",
                           "effect": "label_only" if field == "Pp" else "input_feature"})
    for idx in np.flatnonzero(invalid_names.to_numpy()):
        issues.append({"well": "", "source_row": int(idx + 2), "field": "well", "source_value": "",
                       "reason": "missing_well_name", "effect": "split"})
    if cfg.get('training_selection')=='checked_files':
        cfg['development_wells']=[name for name in frame.well.dropna().unique().tolist() if name]
        cfg['blind_wells']=[]
        if cfg.get('split_method') in ('random_points','leave_one_well_out'):cfg['validation_wells']=[]
    development, blind = set(cfg["development_wells"]), set(cfg["blind_wells"])
    validation = set(cfg.get("validation_wells", []))
    expected = development | blind
    present = set(frame.well.dropna())
    missing_wells = expected - present
    if missing_wells and require_configured_wells:
        raise ValueError(f"配置中的井在文件中不存在: {sorted(missing_wells)}")
    frame["role"] = "unassigned"
    frame.loc[frame.well.isin(development - validation), "role"] = "train"
    frame.loc[frame.well.isin(validation), "role"] = "validation"
    frame.loc[frame.well.isin(blind), "role"] = "blind"
    frame["prediction_valid"] = feature_validity(frame[list(features)].to_numpy(),features) & np.isfinite(frame.Depth) & (frame.Depth>=0)
    frame["supervised_valid"] = frame.prediction_valid & np.isfinite(frame.Pp) & (frame.Pp >= 0) & ~invalid_names
    method = cfg.get("split_method", "well_holdout")
    strategy = {"method": method}
    if method=='leave_one_well_out':
        if len(cfg['development_wells'])<2:raise ValueError('留一井验证至少需要两口参与井')
        heldout=cfg['development_wells'][0]
        frame.loc[frame.well==heldout,'role']='validation'
        strategy.update(preview_only=True,preview_validation_well=heldout,scope='first_fold_preview; actual training runs every well once')
    if method == "random_points":
        # Pool valid development points before splitting; never include blind rows.
        eligible = frame.index[frame.well.isin(development) & frame.supervised_valid].to_numpy()
        seed, ratio = int(project.settings["training"]["seed"]), float(cfg.get("train_ratio", .8))
        indices = np.random.RandomState(seed).permutation(eligible)
        train_count = int(len(indices) * ratio)
        frame.loc[frame.well.isin(development), "role"] = "excluded_invalid"
        frame.loc[indices[:train_count], "role"] = "train"
        frame.loc[indices[train_count:], "role"] = "validation"
        strategy.update(train_ratio=ratio, validation_ratio=1 - ratio, seed=seed,
                        eligible_rows=len(indices), train_rows=train_count, validation_rows=len(indices) - train_count,
                        scope="pooled_valid_development_points", count_rule="floor(train_ratio * eligible_rows)")
    per_well = []
    for name, group in frame.groupby("well", sort=False):
        depths = group.Depth.to_numpy()
        differences = np.diff(depths)
        positive = differences[np.isfinite(differences) & (differences > 0)]
        per_well.append({"well": name, "role": ", ".join(sorted(group.role.unique())), "rows": len(group),
                         "role_rows": {role: int(count) for role, count in group.role.value_counts().items()},
                         "prediction_valid": int(group.prediction_valid.sum()),
                         "supervised_valid": int(group.supervised_valid.sum()),
                         "depth_step_median_m": float(np.median(positive)) if len(positive) else None,
                         "depth_step_max_m": float(np.max(positive)) if len(positive) else None,
                         "duplicate_depth_rows": int(group.Depth.duplicated(keep=False).sum()),
                         "depth_backsteps": int(np.count_nonzero(differences < 0))})
    issues_df = pd.DataFrame(issues, columns=["well", "source_row", "field", "source_value", "reason", "effect"])
    warnings = ["尚未自动平滑或重采样；井震分辨率匹配需要单独验证。"]
    if method == "random_points":
        warnings.append("随机按采样点划分；同一参与井可同时参与训练与验证，验证指标不代表新井泛化能力。"+('训练井由勾选文件决定。' if cfg.get('training_selection')=='checked_files' else '独立盲井不参与划分。'))
    if method=='leave_one_well_out':warnings.append('数据检查仅展示留一井第一折的划分；正式运行将轮流验证每一口参与井。')
    if present - expected or invalid_names.any():
        warnings.append("存在未分配或无井名的记录，已排除训练和评价。")
    if any(w["duplicate_depth_rows"] or w["depth_backsteps"] for w in per_well):
        warnings.append("存在重复深度或深度倒序，应核对后再训练。")
    report = {"source": str(paths[0]) if len(paths)==1 else [str(p) for p in paths], "sources":sources,"source_sha256": source_digest, "rows": len(frame),
              "features": list(features), "canonical_units": {"Depth": "m", "AC": "us/ft", "DEN": "g/cc", "Ip": "g/cc*m/s", "Pp": "MPa"},
              "issue_cells": len(issues_df), "issue_counts": issues_df.groupby("field").size().to_dict(),
              "unassigned_rows": int((frame.role == "unassigned").sum()), "wells": per_well, "warnings": warnings}
    report['canonical_units'].update(feature_units(project.settings,features))
    if library:report['depth_alignment']='exact depth match rounded to 1e-6 m; no implicit interpolation'
    report["split_strategy"] = strategy
    return WellAudit(frame, issues_df, report)


def prepare_data(project: Project) -> PreparedData:
    audit = audit_wells(project)
    features=selected_features(project.settings)
    splits = {role: audit.frame.loc[(audit.frame.role == role) & audit.frame.supervised_valid].copy()
              for role in ("train", "validation", "blind")}
    if project.settings["data"].get("split_method", "well_holdout") == "random_points":
        expected = set(project.settings["data"]["development_wells"])
        valid_wells = set(audit.frame.loc[audit.frame.supervised_valid, "well"])
        if expected - valid_wells:
            raise ValueError("开发井中有整口井缺少有效监督样本")
    for role in ("train", "validation"):
        frame = splits[role]
        if frame.empty:
            raise ValueError(f"{role} 没有有效的监督样本")
    for role in ("train", "validation"):
        expected = set(audit.frame.loc[audit.frame.role == role, "well"].dropna())
        if expected - set(splits[role].well):
            raise ValueError(f"{role} 中有整口井缺少有效监督样本")
    scaler = StandardScaler().fit(splits["train"][list(features)].to_numpy())
    return PreparedData(audit, splits, scaler)
