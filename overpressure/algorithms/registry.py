"""Algorithm capabilities and lazy backend loading, shared by UI and workers."""
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Protocol

import numpy as np


class Predictor(Protocol):
    """Inputs follow manifest feature order and canonical units; invalid rows stay NaN."""
    folder: Path
    metadata: dict
    features: tuple[str, ...]
    feature_units: dict

    def predict(self, features, batch_size: int = 8192) -> np.ndarray: ...


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    title: str
    training_physics: tuple[str, ...] = ()
    trainer: str | None = None
    predictor: str | None = None

    @property
    def available(self):
        return self.trainer is not None and self.predictor is not None

    def require_available(self):
        if not self.available:
            raise ValueError(f'{self.title} 尚未接入，请选择已实现的算法')

    def train(self, project, **kwargs):
        self.require_available()
        module, attribute = self.trainer.rsplit(':', 1)
        return getattr(import_module(module), attribute)(project, **kwargs)

    def load(self, folder, device=None) -> Predictor:
        self.require_available()
        module, attribute = self.predictor.rsplit(':', 1)
        return getattr(import_module(module), attribute)(folder, device=device)


ALGORITHM_SPECS = {
    'mlp': AlgorithmSpec('mlp', 'MLP', ('eaton_soft_loss',),
                         'overpressure.algorithms.mlp_training:train',
                         'overpressure.algorithms.mlp:MLPBundle'),
    'rf': AlgorithmSpec('rf', '随机森林 RF', (),
                        'overpressure.algorithms.tree_training:train', 'overpressure.algorithms.trees:TreeBundle'),
    'xgboost': AlgorithmSpec('xgboost', 'XGBoost', (),
                             'overpressure.algorithms.tree_training:train', 'overpressure.algorithms.trees:TreeBundle'),
}


def get_algorithm(name):
    try:
        return ALGORITHM_SPECS[name]
    except (KeyError, TypeError) as exc:
        raise ValueError(f'未知算法: {name}') from exc


def capabilities():
    return {name: {'planned': not spec.available,
                   'training_physics': list(spec.training_physics)}
            for name, spec in ALGORITHM_SPECS.items()}
