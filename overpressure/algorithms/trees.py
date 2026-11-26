"""Tree model bundles: native XGBoost JSON and pickle-free RF node arrays."""
import json
from pathlib import Path
import numpy as np
from ..config import selected_features
from ..data import sha256_file, feature_validity


def save_forest(model, path):
    arrays = {}
    for i, estimator in enumerate(model.estimators_):
        tree = estimator.tree_
        for name, value in dict(left=tree.children_left, right=tree.children_right,
                                feature=tree.feature, threshold=tree.threshold,
                                value=tree.value[:, 0, 0]).items():
            arrays[f'{i}_{name}'] = value
    arrays['trees'] = np.asarray(len(model.estimators_))
    np.savez_compressed(path, **arrays)


class TreeBundle:
    def __init__(self, folder, device=None):
        self.folder = Path(folder).resolve()
        self.metadata = meta = json.loads((self.folder/'manifest.json').read_text(encoding='utf-8'))
        if meta.get('schema_version') != 1 or meta.get('status') != 'completed' or meta.get('algorithm') not in ('rf','xgboost'):
            raise ValueError('树模型版本、算法或完成状态不受支持')
        self.features = selected_features({'data':{'features':meta['features']}})
        self.feature_units = meta.get('feature_units', {})
        self.device = 'cpu'
        filename = 'forest.npz' if meta['algorithm'] == 'rf' else 'model.json'
        path = self.folder/filename
        if sha256_file(path) != meta['weights_sha256']: raise ValueError('模型权重文件校验失败')
        if meta['algorithm'] == 'rf':
            with np.load(path, allow_pickle=False) as archive:
                self.trees = [{key:archive[f'{i}_{key}'].copy() for key in ('left','right','feature','threshold','value')}
                              for i in range(int(archive['trees']))]
            if not self.trees: raise ValueError('随机森林没有决策树')
            for tree in self.trees:
                n = len(tree['left']); branch = tree['left'] != -1
                if (any(v.shape != (n,) for v in tree.values()) or not np.isfinite(tree['value']).all()
                    or np.any(tree['left'][branch] <= np.flatnonzero(branch))
                    or np.any(tree['right'][branch] <= np.flatnonzero(branch))
                    or np.any(tree['left'][branch] >= n) or np.any(tree['right'][branch] >= n)
                    or np.any(tree['feature'][branch] < 0) or np.any(tree['feature'][branch] >= len(self.features))):
                    raise ValueError('随机森林节点数据无效')
        else:
            import xgboost as xgb
            self.model = xgb.Booster(params={'nthread':4}); self.model.load_model(path)
            if self.model.num_features() != len(self.features): raise ValueError('树模型输入维度不一致')

    def predict(self, features, batch_size=8192):
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.features): raise ValueError(f'预测输入应为 {len(self.features)} 列特征数组')
        if batch_size < 1: raise ValueError('预测批量必须大于零')
        indices = np.flatnonzero(feature_validity(x, self.features))
        result = np.full(len(x), np.nan, dtype=np.float32)
        for start in range(0, len(indices), batch_size):
            rows = indices[start:start+batch_size]
            with np.errstate(over='ignore'): block = x[rows].astype(np.float32)
            if not np.isfinite(block).all(): raise ValueError('输入超出树模型数值范围')
            if self.metadata['algorithm'] == 'rf':
                pred = np.zeros(len(block), dtype=float)
                for tree in self.trees:
                    nodes = np.zeros(len(block), dtype=np.int64)
                    active = np.arange(len(block))
                    while len(active):
                        branch = tree['left'][nodes[active]] != -1; active = active[branch]
                        at = nodes[active]
                        nodes[active] = np.where(block[active, tree['feature'][at]] <= tree['threshold'][at],
                                                 tree['left'][at], tree['right'][at])
                    pred += tree['value'][nodes]
                pred /= len(self.trees)
            else:
                pred = self.model.inplace_predict(block, iteration_range=(0, self.metadata['best_epoch']))
            if not np.isfinite(pred).all(): raise ValueError('树模型预测产生非有限值')
            result[rows] = pred
        return result
