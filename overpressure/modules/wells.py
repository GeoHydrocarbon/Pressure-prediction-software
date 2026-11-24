"""Generic multi-well prediction and optional sparse-pressure validation."""
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from ..config import FEATURES, Project, unit_factor
from ..data import read_table, feature_validity, sha256_file, audit_wells
from ..model import ModelBundle
from ..evaluation import metrics, evaluation_table, MAPE_RULE
from ..well_points import prediction_defaults, match_pressure_points, well_roles, ROLE_LABELS, ROLE_ENGLISH
from ..jobs import emit, save_json, Cancelled
from ..well_io import read_well_files,selected_paths
import re


def read_well_input(path, columns, units, measured=False, features=FEATURES, expected_units=None):
    required = ("well", "Depth", "Pp") if measured else ("well", "Depth", *features)
    paths=path if isinstance(path,(list,tuple)) else [path]
    columns=dict(columns)
    for name in features:columns.setdefault(name,name)
    for name in features:
        if name not in ("Depth","AC","DEN","Ip") and expected_units is not None and units.get(name,"")!=expected_units.get(name,""):raise ValueError(f'曲线 {name} 的单位与模型不一致')
    raw,sources,digest=read_well_files(paths,columns,required=required,reject_duplicate_wells=not measured)
    frame = pd.DataFrame({"source_row": np.arange(len(raw)) + 2})
    frame['source_file']=raw['_source_file'];frame['source_file_row']=raw['_source_file_row']
    frame["well"] = raw[columns["well"]].astype("string").str.strip()
    fields = ("Depth", "Pp") if measured else tuple(dict.fromkeys(("Depth", *features, "Pp")))
    for field in fields:
        original = raw[columns[field]] if columns.get(field) in raw else pd.Series(np.nan, index=raw.index)
        if field == "Pp": frame["reference_supplied"] = original.notna() & original.astype(str).str.strip().ne("")
        frame[field] = pd.to_numeric(original, errors="coerce").to_numpy(dtype=float, na_value=np.nan) * (unit_factor(field, units[field]) if field in ("Depth","AC","DEN","Ip","Pp") else 1.)
    names_valid = frame.well.notna() & frame.well.ne("")
    if not measured:
        frame["prediction_valid"] = feature_validity(frame[list(features)].to_numpy(),features) & names_valid & np.isfinite(frame.Depth) & (frame.Depth>=0)
        frame["supervised_valid"] = frame.prediction_valid & np.isfinite(frame.Pp) & (frame.Pp >= 0)
    frame.attrs.update(sources=sources,source_sha256=digest)
    return frame


def predict_new_wells(project, bundle_path, cancel_file=None):
    def check_cancel():
        if cancel_file and Path(cancel_file).exists(): raise Cancelled("用户取消井预测与验证")
    check_cancel()
    cfg = prediction_defaults(project); bundle = ModelBundle(bundle_path)
    for flag in ("independence_confirmed",):
        if not isinstance(cfg[flag], bool): raise ValueError(f"{flag} 必须为 true 或 false")
    if bundle.metadata.get("purpose") == "smoke_test": raise ValueError("试算模型不能用于正式预测或验证")
    gap = float(cfg["max_gap_m"])
    if not np.isfinite(gap) or gap < 0: raise ValueError("最大匹配间隔必须为非负米数")
    source = selected_paths(project,cfg)
    emit("stage", message="正在读取测井数据并预测；不使用当前开发井/盲井名单筛选新井。")
    features=bundle.features
    library_names=cfg.get('library_wells',[])
    if cfg.get('use_library') and not library_names:raise ValueError('请在井库清单中至少选择一口待预测井')
    if library_names:
        from ..well_library import read_library_frame
        frame,sources,digest=read_library_frame(project,library_names,features,bundle.metadata.get('target_curve','Pp'),require_labels=False,expected_units=bundle.feature_units)
        frame['reference_supplied']=np.isfinite(frame.Pp)
        frame['prediction_valid']=feature_validity(frame[list(features)].to_numpy(),features)
        frame['supervised_valid']=frame.prediction_valid & np.isfinite(frame.Pp) & (frame.Pp>=0)
        frame.attrs.update(sources=sources,source_sha256=digest)
    else:
        frame = read_well_input(source, cfg["columns"], cfg["units"],features=features,expected_units=bundle.feature_units)
    if not frame.prediction_valid.any(): raise ValueError('没有井名和模型所需曲线均有效的预测点，请核对映射、单位与数据')
    frame["Pp_pred"] = bundle.predict(frame[list(features)].to_numpy())
    check_cancel()
    frame.loc[~frame.prediction_valid, "Pp_pred"] = np.nan
    low, high = (np.asarray(bundle.metadata["preprocessing"][key]) for key in ("minimum", "maximum"))
    x = frame[list(features)].to_numpy()
    frame["outside_training_range"] = ((x < low) | (x > high)).any(axis=1) & frame.prediction_valid
    measurement_path = [] if library_names else selected_paths(project,cfg,measured=True)
    if library_names and any(any(c.get('role')=='measurement' for c in well['curves']) for well in project.settings.get('project_data',{}).get('wells',[]) if well['name'] in library_names):
        from ..well_library import read_library_measurements
        points=read_library_measurements(project,library_names)
        measurement_path=['well_library_measurements']
    elif measurement_path:
        points = read_well_input(measurement_path, cfg["measurement_columns"], cfg["measurement_units"], measured=True)
    else:
        # Only supplied labels are measurement points; missing labels remain missing.
        points = frame.loc[frame.reference_supplied, ["source_row",'source_file','source_file_row', "well", "Depth", "Pp"]].copy()
    matched = match_pressure_points(frame, points, gap)
    names = list(dict.fromkeys([*frame.well.dropna().astype(str), *matched.well.dropna().astype(str)]))
    roles = well_roles(bundle.metadata, names, cfg["independence_confirmed"])
    matched["evaluation_role"] = matched.well.map(roles)
    table = evaluation_table(matched)
    present = set(table.loc[table.scope == "well", "well"])
    for name in names:
        if name not in present:
            from ..evaluation import metrics
            table.loc[len(table)] = {"scope": "well", "well": name, **metrics([], [])}
    table["evaluation_role"] = table.well.map(roles).fillna("mixed_or_pooled")
    report = {str(row["well"]): {key: None if pd.isna(row[key]) else int(row[key]) if "count" in key else float(row[key]) for key in ("count", "mae", "rmse", "mape", "r2", "bias", "mape_count", "mape_excluded_zero_count")}
              for _, row in table.loc[table.scope == "well"].iterrows()}
    out = project.resolve(project.settings["output_dir"]) / "well_predictions" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True)
    frame.to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
    matched.to_csv(out / "reference_points.csv", index=False, encoding="utf-8-sig")
    table.to_csv(out / "evaluation_metrics.csv", index=False, encoding="utf-8-sig")
    notes = {"workflow": "sparse_well_validation", "status": "running", "model": str(bundle.folder),"features":list(features),"feature_units":bundle.feature_units, "source_sha256": frame.attrs['source_sha256'],'sources':frame.attrs['sources'],
             "measurement_source_sha256": points.attrs.get('source_sha256') if measurement_path else None,
             "reference_source": "separate_measurements" if measurement_path else "embedded_labels", "settings": cfg,
             "well_roles": roles, "wells": report, "mape_definition": MAPE_RULE,
             "matching_rule": "exact or linear between immediate valid neighbors; no extrapolation; automatic gap = 5 * median positive depth step",
             "point_status_counts": matched.match_status.value_counts().to_dict(), "prediction_rows": len(frame),
             "valid_prediction_rows": int(frame.prediction_valid.sum()), "measurement_rows": len(matched),
             "matched_rows": int(matched.supervised_valid.sum()), "unmatched_rows": int((~matched.supervised_valid).sum())}
    save_json(out / "metrics.json", notes)
    well_dir=out/'per_well';well_dir.mkdir()
    files=[]
    for number,(name,group) in enumerate(frame.groupby('well',sort=False),1):
        check_cancel();stem=f'{number:03d}_'+(re.sub(r'[^\w.-]+','_',str(name)).strip('.')[:100] or 'well')
        group.to_csv(well_dir/(stem+'_predictions.csv'),index=False,encoding='utf-8-sig')
        matched.loc[matched.well==name].to_csv(well_dir/(stem+'_measurements.csv'),index=False,encoding='utf-8-sig')
        files.append(dict(well=str(name),predictions='per_well/'+stem+'_predictions.csv',measurements='per_well/'+stem+'_measurements.csv',rows=len(group),valid_points=int(group.prediction_valid.sum())))
    pd.DataFrame(files).to_csv(out/'well_files.csv',index=False,encoding='utf-8-sig')
    save_json(out / "evaluation_notes.json", {"dataset": "measured_pressure_points", "mape_definition": MAPE_RULE,
              "matching_rule": notes["matching_rule"], "well_roles": roles,
              "scope": "Only successfully matched measurement rows; missing pressure labels are not interpolated"})
    from ..charts import pressure_figure, scatter_figure, save_figure
    try:
        for index, name in enumerate(frame.well.dropna().astype(str).unique(), 1):
            check_cancel()
            save_figure(pressure_figure(frame, name, observations=matched, max_gap_m=gap, source_label=ROLE_ENGLISH[roles[name]]), out / f"pressure_{index}.png")
        if matched.supervised_valid.any():
            check_cancel()
            save_figure(scatter_figure(matched, source_label="Measured pressure points · matched depths"), out / "prediction_vs_reference.png")
        notes["status"] = "completed"; save_json(out / "metrics.json", notes)
    except BaseException as exc:
        notes.update(status="cancelled" if isinstance(exc, (Cancelled, KeyboardInterrupt)) else "failed", error=str(exc))
        save_json(out / "metrics.json", notes); raise
    emit("stage", message=f"预测有效点 {int(frame.prediction_valid.sum())}；实测点 {len(matched)}，匹配 {int(matched.supervised_valid.sum())}；未匹配原因见 reference_points.csv。")
    for name, role in roles.items(): emit("stage", message=f"{name}：{ROLE_LABELS[role]}")
    if not matched.supervised_valid.any(): emit("stage", message="没有成功匹配的有效实测压力点：本次仅输出预测，不给出有效精度评价。")
    emit("completed", path=str(out), metrics=report)
    return out


def predict_wells(project: Project, bundle_path, *, blind_only=False):
    bundle = ModelBundle(bundle_path)
    features=bundle.features
    from copy import deepcopy
    cfg=deepcopy(project.settings);cfg['data'].update(features=list(features),feature_units=bundle.feature_units,target_curve=bundle.metadata.get('target_curve','Pp'))
    project=Project(project.path,cfg)
    if bundle.metadata["purpose"] == "smoke_test":
        raise ValueError("冒烟试算模型仅用于验证链路，不能用于正式井预测或盲井评价")
    current = project.settings["data"]
    if blind_only and set(current["blind_wells"]) != set(bundle.metadata["blind_wells_excluded"]):
        raise ValueError("当前盲井划分与模型记录不一致")
    audit = audit_wells(project, require_labels=blind_only, require_configured_wells=False)
    frame = audit.frame.copy()
    if blind_only:
        frame = frame.loc[frame.well.isin(bundle.metadata["blind_wells_excluded"])].copy()
        if set(bundle.metadata["blind_wells_excluded"]) - set(frame.well):
            raise ValueError("井数据中缺少模型记录的盲井")
    frame["Pp_pred"] = bundle.predict(frame[list(features)].to_numpy())
    if blind_only and any(not (g.supervised_valid & np.isfinite(g.Pp_pred)).any() for _, g in frame.groupby("well")):
        raise ValueError("至少一口盲井没有有效标签与预测配对，无法评价")
    low = np.asarray(bundle.metadata["preprocessing"]["minimum"])
    high = np.asarray(bundle.metadata["preprocessing"]["maximum"])
    x = frame[list(features)].to_numpy()
    frame["outside_training_range"] = ((x < low) | (x > high)).any(axis=1)
    report = {name: metrics(g.Pp.where(g.supervised_valid), g.Pp_pred) for name, g in frame.groupby("well", sort=False)}
    out = project.resolve(project.settings["output_dir"]) / "well_predictions" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True)
    frame.to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
    save_json(out / "metrics.json", {"model": str(bundle.folder), "source_sha256": audit.report["source_sha256"], "blind_only": blind_only, "wells": report})
    evaluation_table(frame).to_csv(out / "evaluation_metrics.csv", index=False, encoding="utf-8-sig")
    from ..charts import pressure_figure, scatter_figure, save_figure
    save_figure(scatter_figure(frame, source_label="Independent blind-well evaluation" if blind_only else "Well prediction · supplied reference labels"), out / "prediction_vs_reference.png")
    save_json(out / "evaluation_notes.json", {"dataset": "blind_wells" if blind_only else "well_prediction", "mape_definition": MAPE_RULE})
    for index, well in enumerate(frame.well.dropna().unique()):
        save_figure(pressure_figure(frame, well), out / f"pressure_{index + 1}.png")
    emit("completed", path=str(out), metrics=report)
    return out
