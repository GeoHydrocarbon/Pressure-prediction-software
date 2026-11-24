"""Business modules and their existing document pages, independent of Qt."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModuleSpec:
    key: str
    title: str
    pages: tuple[int, ...]
    commands: tuple[str, ...] = ()


MODULES = (
    ModuleSpec('training', '模型训练', (1,), ('train', 'cross-validate', 'refit')),
    ModuleSpec('traditional', '传统压力预测', (7,), ('traditional-trend', 'traditional-predict')),
    ModuleSpec('wells', '井预测与验证', (3,), ('predict-wells',)),
    ModuleSpec('seismic', '三维地震预测', (5,), ('predict-seismic',)),
    ModuleSpec('results', '成果查看', (0, 4, 6), ('prepare-seismic-view',)),
)


def module_for_page(index):
    return next((module for module in MODULES if index in module.pages), None)
