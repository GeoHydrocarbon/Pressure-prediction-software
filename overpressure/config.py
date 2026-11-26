from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import yaml


FEATURES = ("Depth", "AC", "DEN", "Ip")
UNITS = {
    "Depth": {"m": 1.0, "ft": 0.3048},
    "AC": {"us/ft": 1.0, "us/m": 0.3048},
    "DEN": {"g/cc": 1.0, "kg/m3": 0.001},
    "Ip": {"g/cc*m/s": 1.0, "kg/m3*m/s": 0.001},
    "Pp": {"MPa": 1.0, "kPa": 0.001, "psi": 0.006894757293168},
}

CANONICAL_UNITS = {key: next(iter(values)) for key, values in UNITS.items()}
RESERVED_FEATURES = {'well','Pp','source_row','source_file','source_file_row','role','prediction_valid','supervised_valid','Pp_pred'}

def selected_features(cfg):
    data=cfg.get('data',cfg)
    features=data.get('features',list(FEATURES))
    if not isinstance(features,(list,tuple)) or not features or len(features)>64 or any(not isinstance(k,str) or not k.strip() or k in RESERVED_FEATURES for k in features) or len(set(features))!=len(features):
        raise ValueError('请选择 1～64 条不重复的输入曲线；压力目标不能同时作为输入')
    target=data.get('target_curve','Pp')
    if target in features:raise ValueError('压力目标曲线不能同时作为模型输入，避免标签泄漏')
    return tuple(features)

def feature_units(cfg,features=None):
    data=cfg.get('data',cfg)
    return {name:CANONICAL_UNITS.get(name,data.get('feature_units',{}).get(name,'')) for name in (features or selected_features(cfg))}

from .algorithms.registry import capabilities

ALGORITHMS = capabilities()


def unit_factor(field: str, unit: str) -> float:
    try:
        return UNITS[field][unit]
    except KeyError as exc:
        raise ValueError(f"不支持的单位: {field}={unit}") from exc


@dataclass(frozen=True)
class Project:
    path: Path
    settings: dict

    def resolve(self, value: str) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (self.path.parent / path).resolve()


def validate_settings(cfg: dict) -> None:
    if cfg.get("schema_version") != 1:
        raise ValueError("项目 schema_version 必须为 1")
    data = cfg["data"]
    for key in ('well_files','well_imports'):
        if key in data and (not isinstance(data[key],list) or any(not isinstance(p,str) or not p.strip() for p in data[key])):raise ValueError(f'{key} 必须为文件路径列表')
    method = data.get("split_method", "well_holdout")
    if method not in ("random_points", "well_holdout",'leave_one_well_out'):
        raise ValueError("数据划分方式必须为 random_points、leave_one_well_out 或 well_holdout")
    ratio = data.get("train_ratio", .8)
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or not 0 < ratio < 1:
        raise ValueError("训练比例必须为 0 与 1 之间的有限数")
    seed = cfg["training"]["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("随机种子必须为 0 至 4294967295 的整数")
    features=selected_features(cfg)
    units=data.get('feature_units',{})
    if not isinstance(units,dict) or any(not isinstance(v,str) for v in units.values()):raise ValueError('曲线单位必须为文本映射')
    for feature in features:
        if feature not in UNITS and not units.get(feature,'').strip():raise ValueError(f'请在导入曲线时填写 {feature} 的单位；无量纲曲线可填写“无量纲”')
    required = {"well", *FEATURES, "Pp"}
    if not required.issubset(data["columns"]) or len(set(data["columns"].values())) != len(data["columns"]):
        raise ValueError("字段映射必须分别包含 well、Depth、AC、DEN、Ip、Pp，且不能重复")
    for field in (*FEATURES, "Pp"):
        unit_factor(field, data["units"][field])
    # Depth type, datum and legacy confirmation flags are optional metadata only.
    for key in ("development_wells", "validation_wells", "blind_wells"):
        values = data.get(key, [])
        if values == [] and (key=='development_wells' and data.get('training_selection') in ('checked_files','well_library') or key == "blind_wells" or key == "validation_wells" and method in ("random_points",'leave_one_well_out')):
            continue
        if not isinstance(values, list) or not values or any(not isinstance(v, str) or not v.strip() for v in values):
            raise ValueError(f"{key} 必须为非空井名列表")
        if len(values) != len(set(values)):
            raise ValueError(f"{key} 中有重复井名")
    development, validation, blind = (set(data.get(k, [])) for k in ("development_wells", "validation_wells", "blind_wells"))
    if development & blind or validation & blind:
        raise ValueError("盲井不能出现在开发井或内部验证井中")
    if method == "well_holdout" and not validation < development:
        raise ValueError("内部验证井必须为开发井的非空真子集")
    name = cfg["algorithm"]["name"]
    if name not in ALGORITHMS:
        raise ValueError(f"未知算法: {name}")
    if name in ('rf', 'xgboost'):
        from .algorithms.tree_settings import parameters
        parameters(cfg)
    physics = cfg["physics"]
    if not isinstance(physics["enabled"], bool):
        raise ValueError("physics.enabled 必须为 true 或 false")
    if not math.isfinite(float(physics["weight"])) or not 0 <= physics["weight"] < 1:
        raise ValueError("物理损失权重必须满足 0 <= weight < 1")
    if physics["enabled"]:
        if not {"Depth","AC"}.issubset(features):raise ValueError('Eaton 物理约束需要把 Depth 和 AC 选为输入曲线')
        if physics["weight"] <= 0:
            raise ValueError("启用物理约束时权重必须大于 0")
        if physics["method"] not in ALGORITHMS[name]["training_physics"]:
            raise ValueError(f"{name} 尚不支持此训练物理约束")
        for key in ("eaton_exponent", "fluid_density", "rho_rock", "gravity"):
            if not math.isfinite(float(physics[key])) or physics[key] <= 0:
                raise ValueError(f"物理参数 {key} 必须为有限正数")
        for key in ("nac_a", "nac_b"):
            if not math.isfinite(float(physics[key])):
                raise ValueError(f"物理参数 {key} 必须为有限数")
        if physics["rho_rock"] <= physics["fluid_density"]:
            raise ValueError("岩石密度必须大于流体密度")
    seismic = cfg["seismic"]
    block = seismic.get("block_traces", 32)
    if isinstance(block, bool) or not isinstance(block, int) or not 1 <= block <= 256:
        raise ValueError("分块道数必须为 1–256 的整数")
    unit_factor("DEN", seismic["density_unit"])
    unit_factor("Ip", seismic["impedance_unit"])
    if seismic["time_convention"] not in ("twt", "owt"):
        raise ValueError("地震时间类型必须为 twt 或 owt")
    if int(seismic["audit_sample_traces"]) < 1:
        raise ValueError("抽样道数必须大于 0")
    for key in ("inline", "crossline", "x", "y", "coordinate_scalar", "coordinate_units"):
        field = seismic["header_fields"][key]
        if not isinstance(field, int) or not 1 <= field <= 240:
            raise ValueError(f"道头位置无效: {key}")


def strip_depth_metadata(cfg: dict) -> dict:
    """Ignore removed legacy depth metadata when opening/saving older projects."""
    cfg["data"].pop("depth_reference", None)
    for key in ("depth_type", "datum", "confirmed", "source", "valid_time_range", "extrapolation"):
        cfg["time_depth"].pop(key, None)
    prediction = cfg.get("well_prediction", {})
    for key in ("depth_reference", "measurements_depth_confirmed"):
        prediction.pop(key, None)
    return cfg


def load_project(path: str | Path) -> Project:
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    strip_depth_metadata(cfg)
    validate_settings(cfg)
    return Project(path, cfg)
