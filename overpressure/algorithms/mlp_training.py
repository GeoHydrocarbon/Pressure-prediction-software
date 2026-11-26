from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .. import __version__
from ..config import FEATURES, Project, selected_features, feature_units
from ..data import audit_wells, prepare_data, sha256_file
from .mlp import MLPBundle as ModelBundle, PointMLP, loss_terms
from ..jobs import Cancelled, emit, save_json
from ..evaluation import metrics, evaluation_table, MAPE_RULE

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def train(project: Project, *, smoke=False, cancel_file=None, _prepared=None, _refit_selection=None):
    cfg = project.settings
    features=selected_features(cfg)
    if cfg['physics']['enabled'] and not {'Depth','AC'}.issubset(features):
        raise ValueError('Eaton 物理约束需要同时选择 Depth 和 AC 输入曲线')
    if cfg["algorithm"]["name"] != "mlp":
        raise ValueError("MLP 训练后端只能处理 mlp 算法配置")
    prepared = _prepared if _prepared is not None else prepare_data(project)
    settings = cfg["training"]
    epochs, batch = int(settings["epochs"]), int(settings["batch_size"])
    if _refit_selection is not None:
        epochs = int(_refit_selection["epochs"])
    lr, patience = float(settings["learning_rate"]), int(settings["patience"])
    if epochs < 1 or batch < 1 or not np.isfinite(lr) or lr <= 0 or patience < 1:
        raise ValueError("训练轮数、批量、学习率和早停耐心必须为正值")
    seed = int(settings["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    splits = {name: frame for name, frame in prepared.splits.items() if name != "blind"}
    if smoke:
        epochs = min(epochs, 3)
        splits = {name: pd.concat([g.sample(n=min(len(g), 128), random_state=seed) for _, g in frame.groupby("well")]) for name, frame in splits.items()}
        # The smoke model must use preprocessing fitted to the same smoke training rows.
        prepared.scaler.fit(splits["train"][list(features)].to_numpy())
    model = PointMLP(cfg["algorithm"]["hidden_sizes"], cfg["algorithm"]["dropout"],input_size=len(features)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loaders = {}
    for role, frame in splits.items():
        raw = frame[list(features)].to_numpy(dtype=np.float32, copy=True)
        scaled = prepared.scaler.transform(raw).astype(np.float32)
        loaders[role] = DataLoader(TensorDataset(torch.from_numpy(scaled), torch.from_numpy(frame.Pp.to_numpy(dtype=np.float32, copy=True)), torch.from_numpy(raw)), batch_size=batch, shuffle=role == "train")
    run = project.resolve(cfg["output_dir"]) / "models" / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ("_smoke" if smoke else "_mlp"))
    run.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "software_version": __version__, "status": "running", "algorithm": "mlp",
                "display_name": f"MLP · {datetime.now():%Y-%m-%d %H:%M:%S.%f}",
                "purpose": "smoke_test" if smoke else "final_refit" if _refit_selection else "training", "created_at": datetime.now().astimezone().isoformat(),
                "features": list(features), "feature_units":feature_units(cfg,features), "target_curve":cfg["data"].get("target_curve","Pp"), "architecture": cfg["algorithm"], "project": cfg,
                "source_sha256": prepared.audit.report["source_sha256"],
                "split": {role: {"wells": list(frame.well.unique()), "rows": len(frame)} for role, frame in splits.items()},
                "blind_wells_excluded": cfg["data"]["blind_wells"], "selection_metric": "internal_validation_data_mse",
                "physics": cfg["physics"], "device": str(device), "python": sys.version, "torch": str(torch.__version__),
                "preprocessing": {"mean": prepared.scaler.mean_.tolist(), "scale": prepared.scaler.scale_.tolist(),
                                  "fit_wells": list(splits["train"].well.unique()), "fit_rows": len(splits["train"]),
                                  "minimum": splits["train"][list(features)].min().tolist(),
                                  "maximum": splits["train"][list(features)].max().tolist()}}
    if _refit_selection:
        manifest.update(refit_selection=_refit_selection, selection_metric="fixed_epochs_from_development_cv", validation_used=False)
    manifest["split_strategy"] = ({"method": "all_development_refit"} if _refit_selection else
                                  {"method": "well_holdout"} if _prepared is not None else prepared.audit.report["split_strategy"])
    manifest["split_strategy"] = dict(manifest["split_strategy"], smoke_subsampled=smoke)
    assignments = pd.concat([frame[[key for key in ('source_row','source_file','source_file_row','well') if key in frame]].assign(role=role) for role, frame in splits.items()], ignore_index=True)
    assignments.to_csv(run / "split_assignments.csv", index=False, encoding="utf-8-sig")
    save_json(run / "manifest.json", manifest)
    prepared.audit.issues.to_csv(run / "data_issues.csv", index=False, encoding="utf-8-sig")
    emit("run_started", path=str(run), device=str(device), smoke=smoke)
    emit("stage", message=f"划分方式：{manifest['split_strategy']['method']}；训练 {len(splits['train'])} 点；验证 {len(splits.get('validation', []))} 点；盲井不参与。")
    history, best, stale = [], float("inf"), 0
    try:
        for epoch in range(1, epochs + 1):
            record = {"epoch": epoch}
            for role, loader in loaders.items():
                model.train(role == "train")
                totals, count = np.zeros(3), 0
                for x, y, raw in loader:
                    if cancel_file and Path(cancel_file).exists():
                        raise Cancelled("用户取消训练")
                    x, y, raw = x.to(device), y.to(device), raw.to(device)
                    with torch.set_grad_enabled(role == "train"):
                        total, data, physical = loss_terms(model(x), y, raw, cfg["physics"],features)
                        if role == "train":
                            optimizer.zero_grad(set_to_none=True)
                            total.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                            optimizer.step()
                    totals += np.array([total.item(), data.item(), physical.item()]) * len(x)
                    count += len(x)
                for name, value in zip(("total", "data", "physics"), totals / count):
                    record[f"{role}_{name}"] = float(value)
            history.append(record)
            pd.DataFrame(history).to_csv(run / "history.csv", index=False)
            if _refit_selection or record["validation_data"] < best:
                best, stale = record.get("validation_data", record["train_data"]), 0
                torch.save(model.state_dict(), run / "weights.tmp")
                (run / "weights.tmp").replace(run / "weights.pt")
                if _refit_selection:
                    manifest.update(last_epoch=epoch)
                else:
                    manifest.update(best_epoch=epoch, best_validation_data_mse=best)
            else:
                stale += 1
            emit("epoch", **record, epochs=epochs)
            if not _refit_selection and stale >= patience:
                break
        manifest.update(status="completed", weights_sha256=sha256_file(run / "weights.pt"), epochs_completed=len(history))
        save_json(run / "manifest.json", manifest)
        bundle = ModelBundle(run, device=str(device))
        if "validation" in splits:
            validation = splits["validation"]
            pred = bundle.predict(validation[list(features)].to_numpy())
            manifest["internal_validation_metrics"] = metrics(validation.Pp, pred)
            predictions = validation.copy(); predictions["Pp_pred"] = pred
            predictions.to_csv(run / "validation_predictions.csv", index=False, encoding="utf-8-sig")
            evaluation_table(predictions).to_csv(run / "evaluation_metrics.csv", index=False, encoding="utf-8-sig")
        from ..charts import training_figure, scatter_figure, save_figure
        save_figure(training_figure(pd.DataFrame(history)), run / "training_curve.png")
        if "validation" in splits:
            save_figure(scatter_figure(predictions, source_label="Internal validation" + (" · SMOKE TEST" if smoke else "")), run / "prediction_vs_reference.png")
            save_json(run / "evaluation_notes.json", {"dataset": "internal_validation", "mape_definition": MAPE_RULE, "aggregation": "pooled valid points, not independent blind wells"})
        save_json(run / "manifest.json", manifest)
        if _refit_selection:
            emit("completed", artifact="final_model", path=str(run), metrics={"epochs": len(history), "training_rows": len(splits["train"])}, smoke=smoke)
        else:
            emit("completed", path=str(run), validation=manifest["internal_validation_metrics"], smoke=smoke)
    except BaseException as exc:
        manifest.update(status="cancelled" if isinstance(exc, (Cancelled, KeyboardInterrupt)) else "failed", error=str(exc))
        save_json(run / "manifest.json", manifest)
        raise
    return run

