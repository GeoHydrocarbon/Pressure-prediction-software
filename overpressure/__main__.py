from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import yaml

from . import __version__
from .config import ALGORITHMS, FEATURES, load_project, selected_features
from .data import prepare_data
from .time_depth import TimeDepth


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def audit(project_path, wells_only=False) -> tuple[Path, dict]:
    project = load_project(project_path)
    if not wells_only:TimeDepth(project)
    print("正在检查井数据与训练/验证划分……", flush=True)
    prepared = prepare_data(project)
    cfg = project.settings
    features=selected_features(cfg)
    out = project.resolve(cfg["output_dir"]) / "data_checks" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    report = {"software_version": __version__, "project": str(project.path), "status": "running",
              "well_data": prepared.audit.report, "algorithm_capabilities": ALGORITHMS,
              "split_strategy": prepared.audit.report["split_strategy"],
              "warnings": ["数据检查不等于正式训练或精度验收。", "验证集用于早停和选参；最终新井能力应看独立盲井评价。"]}
    write_json(out / "report.json", report)
    (out / "project_snapshot.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    prepared.audit.issues.to_csv(out / "invalid_cells.csv", index=False, encoding="utf-8-sig")
    prepared.audit.frame[["source_row", "well", "role", "prediction_valid", "supervised_valid"]].to_csv(out / "row_assignments.csv", index=False, encoding="utf-8-sig")
    split = {role: {"wells": list(frame.well.unique()), "valid_supervised_rows": len(frame)} for role, frame in prepared.splits.items()}
    write_json(out / "split_manifest.json", split)
    train_x = prepared.splits["train"][list(features)].to_numpy()
    write_json(out / "preprocessing_preview.json", {
        "purpose": "数据检查预览；正式模型需在自身训练过程中保存对应参数",
        "fit_role": "train", "fit_wells": split["train"]["wells"], "fit_rows": len(train_x), "features": list(features),
        "mean": prepared.scaler.mean_.tolist(), "scale": prepared.scaler.scale_.tolist(),
        "training_min": np.min(train_x, axis=0).tolist(), "training_max": np.max(train_x, axis=0).tolist()})
    for role, item in split.items():
        print(f"{role}: {', '.join(item['wells'])}，有效监督样本 {item['valid_supervised_rows']}", flush=True)
    print(f"问题单元格: {len(prepared.audit.issues)}；详细记录: {out}", flush=True)
    try:
        if not wells_only:
            from .seismic import inspect_seismic
            print("正在逐道核对 SEG-Y 几何和采样，并抽样检查属性……", flush=True)
            report["seismic"] = inspect_seismic(project)
        else:
            report["seismic"] = {"status": "not_checked"}
        report["status"] = "completed" if wells_only or report["seismic"]["passed"] else "failed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
    write_json(out / "report.json", report)
    print(f"检查状态: {report['status']}；报告: {out / 'report.json'}", flush=True)
    return out, report


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="超压预测软件核心：逐点数据检查")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("audit", help="检查井数据、划分、单位和 SEG-Y")
    check.add_argument("--project", default="project.yaml")
    check.add_argument("--wells-only", action="store_true")
    fit = sub.add_parser("train", help="按所选算法训练模型")
    fit.add_argument("--project", default="project.yaml")
    fit.add_argument("--smoke", action="store_true", help="仅做小样本三轮试算，禁止成果用于正式预测")
    fit.add_argument("--cancel-file")
    predict = sub.add_parser("predict-wells", help="通用多井预测、稀疏压力点验证或旧版盲井评价")
    predict.add_argument("--project", default="project.yaml")
    predict.add_argument("--bundle", required=True)
    predict.add_argument("--blind-only", action="store_true")
    predict.add_argument("--generic", action="store_true", help="使用独立的 well_prediction 输入和可选实测点")
    predict.add_argument("--cancel-file")
    cv = sub.add_parser("cross-validate", help="仅对开发井做留一井交叉验证")
    cv.add_argument("--project", default="project.yaml")
    cv.add_argument("--smoke", action="store_true")
    cv.add_argument("--cancel-file")
    refit = sub.add_parser("refit", help="按验证轮数使用全部开发井最终重训")
    refit.add_argument("--project", default="project.yaml")
    refit.add_argument("--cv", required=True)
    refit.add_argument("--smoke", action="store_true")
    refit.add_argument("--cancel-file")
    view = sub.add_parser("prepare-seismic-view", help="校验地震成果并准备切片")
    view.add_argument("--project", default="project.yaml")
    view.add_argument("--result", required=True)
    view.add_argument("--cancel-file")
    seismic = sub.add_parser("predict-seismic", help="自定义线号范围或全体 SEG-Y 压力预测")
    seismic.add_argument("--project", default="project.yaml")
    seismic.add_argument("--bundle", required=True)
    seismic.add_argument("--trace-start", type=int, default=0)
    seismic.add_argument("--trace-count", type=int, default=32)
    seismic.add_argument("--trace-stride", type=int, default=1)
    seismic.add_argument('--inline-start', type=int)
    seismic.add_argument('--inline-end', type=int)
    seismic.add_argument('--crossline-start', type=int)
    seismic.add_argument('--crossline-end', type=int)
    seismic.add_argument("--cancel-file")
    seismic.add_argument("--full", action="store_true", help="预测全部输入道")
    seismic.add_argument("--block-traces", type=int, default=None)

    for command,help_text in [('traditional-trend','正常压实趋势两点标定'),('traditional-predict','独立 Eaton 井压力预测')]:
        traditional=sub.add_parser(command,help=help_text)
        traditional.add_argument('--project',default='project.yaml')
        traditional.add_argument('--cancel-file')
    args = parser.parse_args()
    try:
        if args.command == "audit":
            _, report = audit(args.project, args.wells_only)
            return 0 if report["status"] == "completed" else 1
        if args.command == "prepare-seismic-view":
            from .volume_reader import prepare_index
            path = prepare_index(args.result, args.cancel_file)
            print(json.dumps({"event": "completed", "artifact": "seismic_index", "path": str(path)}, ensure_ascii=False), flush=True)
            return 0
        if args.command in ('traditional-trend','traditional-predict'):
            from .traditional import analyze_trend,predict_eaton
            run=analyze_trend if args.command=='traditional-trend' else predict_eaton
            run(load_project(args.project),cancel_file=args.cancel_file)
            return 0
        from .modules.training import train
        from .modules.wells import predict_wells
        project = load_project(args.project)
        if args.command == "train":
            train(project, smoke=args.smoke, cancel_file=args.cancel_file)
        elif args.command == "cross-validate":
            from .modules.training import cross_validate
            cross_validate(project, smoke=args.smoke, cancel_file=args.cancel_file)
        elif args.command == "refit":
            from .modules.training import refit
            refit(project, args.cv, smoke=args.smoke, cancel_file=args.cancel_file)
        elif args.command == "predict-seismic":
            from .modules.seismic import predict
            bounds = (args.inline_start,args.inline_end,args.crossline_start,args.crossline_end)
            if any(v is not None for v in bounds) and any(v is None for v in bounds):
                raise ValueError('请完整填写 Inline、Crossline 起止线号')
            predict(project, args.bundle, full=args.full,
                    line_range=bounds if any(v is not None for v in bounds) else None,
                    trace_start=args.trace_start, trace_count=args.trace_count,
                    trace_stride=args.trace_stride, cancel_file=args.cancel_file,
                    block_traces=args.block_traces)
        else:
            if args.generic:
                if args.blind_only: raise ValueError("通用预测与旧版盲井筛选不能同时使用")
                from .modules.wells import predict_new_wells
                predict_new_wells(project, args.bundle, cancel_file=args.cancel_file)
            else: predict_wells(project, args.bundle, blind_only=args.blind_only)
        return 0
    except Exception as exc:
        print(f"任务失败: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
