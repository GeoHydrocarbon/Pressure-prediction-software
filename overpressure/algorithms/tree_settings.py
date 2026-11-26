"""Tree defaults and validation shared by desktop and training workers."""
import math

DEFAULTS = {
    'rf': dict(n_estimators=300, max_depth=0, min_samples_split=2,
               min_samples_leaf=1, max_features=1.0, bootstrap=True, n_jobs=4),
    'xgboost': dict(n_estimators=500, max_depth=6, learning_rate=.05,
                    subsample=.8, colsample_bytree=.8, min_child_weight=1.,
                    reg_alpha=0., reg_lambda=1., early_stopping_rounds=20, n_jobs=4),
}


def parameters(cfg, name=None):
    name = name or cfg['algorithm']['name']
    values = dict(DEFAULTS[name], **cfg['algorithm'].get(name, {}))
    for key, value in values.items():
        if key not in DEFAULTS[name]:
            raise ValueError(f'{name}: 未知参数 {key}')
        if isinstance(DEFAULTS[name][key], bool):
            if not isinstance(value, bool): raise ValueError(f'{key} 必须为布尔值')
        elif isinstance(DEFAULTS[name][key], int):
            lower = 0 if key in ('max_depth', 'early_stopping_rounds') else 2 if key == 'min_samples_split' else 1
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= 100000:
                raise ValueError(f'{key} 必须为 {lower}～100000 的整数')
            if name == 'xgboost' and key == 'max_depth' and value < 1:
                raise ValueError('XGBoost 最大深度必须大于零')
            if key == 'n_jobs' and value > 64: raise ValueError('并行线程数必须为 1～64')
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{key} 必须为有限数值')
            if key in ('learning_rate','subsample','colsample_bytree','max_features'):
                if not 0 < value <= 1: raise ValueError(f'{key} 必须满足 0 < 值 <= 1')
            elif value < 0: raise ValueError(f'{key} 不能为负数')
    return values


def iteration_limit(cfg):
    name = cfg['algorithm']['name']
    return cfg['training']['epochs'] if name == 'mlp' else parameters(cfg)['n_estimators']
