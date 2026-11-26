"""Model display information, independent of its stable on-disk identity."""
import json
from pathlib import Path


def model_info(path):
    folder = Path(path)
    try:
        meta = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    except (OSError, ValueError): meta = {}
    algorithm = {'mlp':'MLP','rf':'RF','xgboost':'XGBoost'}.get(meta.get('algorithm'), '模型')
    name = meta.get('display_name') or (algorithm+' · '+meta['created_at'][:19].replace('T',' ') if meta.get('created_at') else folder.name)
    return dict(name=name, algorithm=algorithm, smoke=meta.get('purpose')=='smoke_test')
