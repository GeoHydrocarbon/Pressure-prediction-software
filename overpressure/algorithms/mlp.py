from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..config import FEATURES, selected_features
from ..data import feature_validity, sha256_file


class PointMLP(nn.Module):
    def __init__(self, hidden_sizes=(256, 64), dropout=0.15, input_size=None):
        super().__init__()
        if not hidden_sizes or any(not isinstance(n, int) or n < 1 for n in hidden_sizes):
            raise ValueError("MLP 隐藏层必须为正整数列表")
        if not 0 <= dropout < 1:
            raise ValueError("dropout 必须在 [0,1) 内")
        layers = []
        self.input_size = len(FEATURES) if input_size is None else int(input_size)
        if not 1 <= self.input_size <= 64:raise ValueError('模型输入维数必须为 1～64')
        last = self.input_size
        for width in hidden_sizes:
            layers.extend((nn.Linear(last, width), nn.ReLU(), nn.Dropout(dropout)))
            last = width
        layers.append(nn.Linear(last, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.input_size:
            raise ValueError(f'逐点 MLP 输入必须为 [点数, {self.input_size}]，不接受时间窗口')
        return self.network(x).squeeze(-1)


def eaton_pressure(raw: torch.Tensor, physics: dict):
    depth, ac = raw[:, 0], raw[:, 1]
    nac = torch.exp(physics["nac_a"] * depth + physics["nac_b"])
    pv = physics["rho_rock"] * physics["gravity"] * depth / 1000.0
    pw = physics["fluid_density"] * physics["gravity"] * depth / 1000.0
    result = pv - (pv - pw) * (nac / ac).pow(physics["eaton_exponent"])
    if not torch.isfinite(result).all():
        raise ValueError("Eaton 计算产生非有限值，请核对单位和物理参数")
    return result


def loss_terms(prediction, label, raw, physics, features=FEATURES):
    data = (prediction - label).square().mean()
    if physics["enabled"]:
        physical = (prediction - eaton_pressure(raw[:, [features.index("Depth"),features.index("AC")]], physics)).square().mean()
        total = (1 - physics["weight"]) * data + physics["weight"] * physical
    else:
        physical = data.new_zeros(())
        total = data
    if not torch.isfinite(total):
        raise ValueError("训练损失非有限，请检查数据和参数")
    return total, data, physical


class MLPBundle:
    def __init__(self, folder: str | Path, device=None):
        self.folder = Path(folder).resolve()
        self.metadata = json.loads((self.folder / "manifest.json").read_text(encoding="utf-8"))
        meta = self.metadata
        if meta.get("schema_version") != 1 or meta.get("algorithm") != "mlp" or meta.get("status") != "completed":
            raise ValueError("模型包版本、算法或完成状态不受支持")
        self.features = selected_features({'data':{'features':meta['features']}})
        self.feature_units = meta.get('feature_units', {})
        if sha256_file(self.folder / "weights.pt") != meta["weights_sha256"]:
            raise ValueError("模型权重文件校验失败")
        pre = meta["preprocessing"]
        self.mean = np.asarray(pre["mean"], dtype=float)
        self.scale = np.asarray(pre["scale"], dtype=float)
        if self.mean.shape != (len(self.features),) or self.scale.shape != (len(self.features),) or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise ValueError("模型标准化参数无效")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        architecture = meta["architecture"]
        self.model = PointMLP(architecture["hidden_sizes"], architecture["dropout"],input_size=len(self.features)).to(self.device)
        self.model.load_state_dict(torch.load(self.folder / "weights.pt", map_location=self.device, weights_only=True))
        self.model.eval()

    def predict(self, features, batch_size=8192):
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.features):
            raise ValueError(f'预测输入必须为二维 {len(self.features)} 特征数组，顺序为 {self.features}')
        if batch_size < 1:
            raise ValueError("预测批量必须大于零")
        valid = feature_validity(x,self.features)
        result = np.full(len(x), np.nan, dtype=np.float32)
        indices = np.flatnonzero(valid)
        with torch.inference_mode():
            for start in range(0, len(indices), batch_size):
                rows = indices[start:start + batch_size]
                scaled = ((x[rows] - self.mean) / self.scale).astype(np.float32)
                if not np.isfinite(scaled).all():
                    raise ValueError("标准化后出现非有限值，输入超出数值范围")
                pred = self.model(torch.from_numpy(scaled).to(self.device)).cpu().numpy()
                if not np.isfinite(pred).all():
                    raise ValueError("模型预测产生非有限值")
                result[rows] = pred
        return result
