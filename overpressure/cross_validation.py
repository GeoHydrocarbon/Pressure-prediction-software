from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import FEATURES, Project, selected_features, feature_units
from .data import PreparedData, WellAudit, audit_wells, development_fingerprint
from .modules.training import train
from .jobs import Cancelled, emit, save_json
from .evaluation import metrics


def cross_validate(project: Project, *, smoke=False, cancel_file=None):
    """All development wells are held out once. Blind wells never enter any fold."""
    audit = audit_wells(project)
    cfg = project.settings
    features=selected_features(cfg)
    wells = cfg["data"]["development_wells"]
    if len(wells) < 2:
        raise ValueError("留一井验证至少需要两口开发井")
    out = project.resolve(cfg["output_dir"]) / "cross_validation" / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ("_smoke" if smoke else ""))
    out.mkdir(parents=True)
    manifest = {"status": "running", "purpose": "smoke_test" if smoke else "cross_validation", "project": cfg,
                "development_sha256": development_fingerprint(audit.frame, wells,features),
                "source_sha256": audit.report["source_sha256"], "blind_wells_excluded": cfg["data"]["blind_wells"],
                "folds": [], "aggregation": "pooled point metrics and unweighted macro average of well metrics"}
    save_json(out / "summary.json", manifest)
    predictions, rows = [], []
    try:
        for fold, heldout in enumerate(wells, 1):
            if cancel_file and Path(cancel_file).exists(): raise Cancelled("用户取消留一井验证")
            fold_cfg = deepcopy(cfg); fold_cfg["data"]["validation_wells"] = [heldout]
            fold_cfg["data"]["split_method"] = "well_holdout"
            fold_cfg["output_dir"] = str(out / f"fold_{fold:02d}")
            fold_project = Project(project.path, fold_cfg)
            frame = audit.frame.copy()
            frame["role"] = "unassigned"
            frame.loc[frame.well.isin(set(wells) - {heldout}), "role"] = "train"
            frame.loc[frame.well == heldout, "role"] = "validation"
            frame.loc[frame.well.isin(cfg["data"]["blind_wells"]), "role"] = "blind"
            splits = {role: frame.loc[(frame.role == role) & frame.supervised_valid].copy() for role in ("train", "validation", "blind")}
            if splits["validation"].empty or splits["train"].empty:
                raise ValueError(f"{heldout}: 训练或验证集没有有效监督点")
            train_wells = set(wells) - {heldout}
            if set(splits["train"].well) != train_wells:
                raise ValueError("有开发井不含有效监督点")
            fold_audit = WellAudit(frame, audit.issues, deepcopy(audit.report))
            prepared = PreparedData(fold_audit, splits, StandardScaler().fit(splits["train"][list(features)].to_numpy()))
            emit("fold_started", fold=fold, folds=len(wells), validation_well=heldout, path=str(out))
            model_path = train(fold_project, smoke=smoke, cancel_file=cancel_file, _prepared=prepared)
            model_meta = json.loads((model_path / "manifest.json").read_text(encoding="utf-8"))
            pred = pd.read_csv(model_path / "validation_predictions.csv", encoding="utf-8-sig")
            pred["fold"] = fold; predictions.append(pred)
            row = {"fold": fold, "validation_well": heldout, "best_epoch": model_meta["best_epoch"],
                   **model_meta["internal_validation_metrics"], "model_path": str(model_path)}
            rows.append(row); manifest["folds"] = rows
            pd.DataFrame(rows).to_csv(out / "folds.csv", index=False, encoding="utf-8-sig")
            save_json(out / "summary.json", manifest)
        combined = pd.concat(predictions, ignore_index=True)
        if combined.source_row.duplicated().any() or set(combined.well) != set(wells):
            raise ValueError("留一井输出出现重复行或缺失开发井")
        combined.to_csv(out / "oof_predictions.csv", index=False, encoding="utf-8-sig")
        manifest.update(status="completed", pooled_metrics=metrics(combined.Pp, combined.Pp_pred),
                        macro_metrics={key: float(np.mean([row[key] for row in rows])) for key in ("mae", "rmse", "bias")},
                        suggested_refit_epochs=int(np.median([row["best_epoch"] for row in rows])))
        from .charts import folds_figure, scatter_figure, save_figure
        from .evaluation import evaluation_table, MAPE_RULE
        save_figure(folds_figure(pd.DataFrame(rows)), out / "fold_errors.png")
        save_figure(scatter_figure(combined, source_label="Leave-one-well-out" + (" · SMOKE TEST" if smoke else "")), out / "prediction_vs_reference.png")
        evaluation_table(combined).to_csv(out / "evaluation_metrics.csv", index=False, encoding="utf-8-sig")
        save_json(out / "evaluation_notes.json", {"dataset": "development_well_oof", "mape_definition": MAPE_RULE})
        save_json(out / "summary.json", manifest)
        emit("completed", artifact="cross_validation", path=str(out), smoke=smoke, metrics=manifest["pooled_metrics"])
    except BaseException as exc:
        manifest.update(status="cancelled" if isinstance(exc, (Cancelled, KeyboardInterrupt)) else "failed", error=str(exc))
        save_json(out / "summary.json", manifest)
        raise
    return out
