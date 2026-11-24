"""Algorithm-independent model loading; legacy MLP imports remain supported."""
from pathlib import Path
import json

from .algorithms.registry import get_algorithm


class ModelBundle:
    def __init__(self, folder, device=None):
        folder = Path(folder).resolve()
        metadata = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
        if metadata.get('schema_version') != 1 or metadata.get('status') != 'completed':
            raise ValueError('模型包版本或完成状态不受支持')
        self._backend = get_algorithm(metadata.get('algorithm')).load(folder, device=device)

    def __getattr__(self, name):
        return getattr(self._backend, name)

    def predict(self, features, batch_size=8192):
        return self._backend.predict(features, batch_size=batch_size)


def __getattr__(name):
    # Keep legacy callers working without importing torch into UI-only processes.
    if name in ('PointMLP', 'eaton_pressure', 'loss_terms'):
        from .algorithms import mlp
        return getattr(mlp, name)
    raise AttributeError(name)
