"""RF/XGBoost training using the same audited splits and outputs as MLP."""
from datetime import datetime
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from .. import __version__
from ..config import selected_features, feature_units
from ..data import prepare_data, sha256_file
from ..evaluation import metrics, evaluation_table, MAPE_RULE
from ..jobs import Cancelled, emit, save_json
from .tree_settings import parameters
from .trees import TreeBundle, save_forest


def train(project, *, smoke=False, cancel_file=None, _prepared=None, _refit_selection=None):
    cfg = project.settings; name = cfg['algorithm']['name']; params = parameters(cfg)
    if cfg['physics']['enabled']: raise ValueError('RF 和 XGBoost 暂不支持物理约束')
    features = selected_features(cfg); seed = cfg['training']['seed']
    prepared = _prepared if _prepared is not None else prepare_data(project)
    splits = {key:value.copy() for key,value in prepared.splits.items() if key != 'blind'}
    rounds = int(_refit_selection['epochs']) if _refit_selection else params['n_estimators']
    if smoke:
        rounds = min(rounds, 8)
        splits = {key:pd.concat([g.sample(n=min(len(g),128), random_state=seed) for _,g in frame.groupby('well')])
                  for key,frame in splits.items()}
    x = {key:frame[list(features)].to_numpy(dtype=np.float32) for key,frame in splits.items()}
    y = {key:frame.Pp.to_numpy(dtype=np.float32) for key,frame in splits.items()}
    run = project.resolve(cfg['output_dir'])/'models'/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+('_smoke_' if smoke else '_')+name)
    run.mkdir(parents=True, exist_ok=False)
    meta = dict(schema_version=1, software_version=__version__, algorithm=name, status='running',
                purpose='smoke_test' if smoke else 'final_refit' if _refit_selection else 'training',
                display_name=f"{'RF' if name == 'rf' else 'XGBoost'} · {datetime.now():%Y-%m-%d %H:%M:%S.%f}",
                created_at=datetime.now().astimezone().isoformat(), features=list(features), feature_units=feature_units(cfg,features),
                target_curve=cfg['data'].get('target_curve','Pp'), architecture=dict(params), project=cfg, physics=dict(cfg['physics']),
                source_sha256=prepared.audit.report['source_sha256'], blind_wells_excluded=cfg['data']['blind_wells'],
                split={key:dict(wells=list(frame.well.unique()), rows=len(frame)) for key,frame in splits.items()},
                split_strategy=dict(method='all_development_refit' if _refit_selection else 'well_holdout') if _prepared else prepared.audit.report['split_strategy'],
                selection_metric='fixed_iterations_from_development_cv' if _refit_selection else 'fixed_tree_count' if name == 'rf' else 'internal_validation_data_mse',
                device='cpu', python=sys.version,
                preprocessing=dict(kind='none', minimum=splits['train'][list(features)].min().tolist(),
                    maximum=splits['train'][list(features)].max().tolist(), fit_wells=list(splits['train'].well.unique()), fit_rows=len(splits['train'])))
    if _refit_selection: meta.update(refit_selection=_refit_selection, validation_used=False)
    save_json(run/'manifest.json', meta)
    pd.concat([frame[[k for k in ('source_row','source_file','source_file_row','well') if k in frame]].assign(role=key)
               for key,frame in splits.items()],ignore_index=True).to_csv(run/'split_assignments.csv',index=False,encoding='utf-8-sig')
    prepared.audit.issues.to_csv(run/'data_issues.csv',index=False,encoding='utf-8-sig')
    emit('run_started',path=str(run),device='cpu',smoke=smoke)
    emit('stage',message=f"{name.upper()}：训练 {len(x['train'])} 点；盲井不参与；无物理约束。")
    history = []
    def check_cancel():
        if cancel_file and Path(cancel_file).exists(): raise Cancelled('用户取消训练')
    def record(iteration, scores):
        check_cancel()
        row = dict(epoch=iteration, train_data=float(scores['train']), train_total=float(scores['train']), train_physics=0.,
                   iteration_unit='trees' if name == 'rf' else 'boosting_rounds')
        if 'validation' in scores: row.update(validation_data=float(scores['validation']), validation_total=float(scores['validation']), validation_physics=0.)
        history.append(row); emit('epoch',**row,epochs=rounds)
    try:
        check_cancel()
        if name == 'rf':
            import sklearn
            from sklearn.ensemble import RandomForestRegressor
            meta['sklearn'] = sklearn.__version__
            options = {k:v for k,v in params.items() if k != 'n_estimators'}
            options['max_depth'] = options['max_depth'] or None
            model = RandomForestRegressor(n_estimators=1, warm_start=True, random_state=seed, **options)
            for count in range(min(25,rounds), rounds+25, 25):
                check_cancel(); count = min(count,rounds); model.set_params(n_estimators=count); model.fit(x['train'],y['train'])
                record(count,{key:np.mean((model.predict(value)-y[key])**2) for key,value in x.items()})
                if count == rounds: break
            best = rounds; save_forest(model,run/'forest.npz'); weights = run/'forest.npz'
        else:
            try: import xgboost as xgb
            except ImportError as exc: raise RuntimeError('计算环境缺少 XGBoost，请在后台 Python 环境安装 requirements.txt 中的依赖') from exc
            meta['xgboost'] = xgb.__version__
            class Progress(xgb.callback.TrainingCallback):
                def after_iteration(self, model, epoch, evals_log):
                    record(epoch+1,{key:values['rmse'][-1]**2 for key,values in evals_log.items()}); return False
            options = {k:v for k,v in params.items() if k not in ('n_estimators','early_stopping_rounds','n_jobs','learning_rate')}
            options.update(objective='reg:squarederror',eval_metric='rmse',tree_method='hist',eta=params['learning_rate'],seed=seed,nthread=params['n_jobs'])
            matrices = {key:xgb.DMatrix(x[key],label=y[key]) for key in ('train','validation') if key in x}
            stopping = params['early_stopping_rounds'] if not _refit_selection and 'validation' in x else 0
            model = xgb.train(options,matrices['train'],num_boost_round=rounds,evals=list((v,k) for k,v in matrices.items()),
                              early_stopping_rounds=stopping or None,verbose_eval=False,callbacks=[Progress()])
            best = model.best_iteration+1 if stopping else rounds
            model = model[:best]; weights = run/'model.json'; model.save_model(weights)
        check_cancel()
        meta.update(status='completed', best_epoch=best, epochs_completed=len(history) if name == 'xgboost' else rounds,
                    weights_sha256=sha256_file(weights))
        save_json(run/'manifest.json',meta)
        bundle = TreeBundle(run)
        pd.DataFrame(history).to_csv(run/'history.csv',index=False,encoding='utf-8-sig')
        from ..charts import training_figure, scatter_figure, save_figure
        figure = training_figure(pd.DataFrame(history)); figure.axes[0].set_xlabel('Number of trees' if name == 'rf' else 'Boosting round')
        save_figure(figure,run/'training_curve.png')
        if 'validation' in splits:
            predictions = splits['validation'].copy(); predictions['Pp_pred'] = bundle.predict(x['validation'])
            meta['internal_validation_metrics'] = metrics(predictions.Pp,predictions.Pp_pred)
            predictions.to_csv(run/'validation_predictions.csv',index=False,encoding='utf-8-sig')
            evaluation_table(predictions).to_csv(run/'evaluation_metrics.csv',index=False,encoding='utf-8-sig')
            save_figure(scatter_figure(predictions,source_label='Internal validation'+(' · SMOKE TEST' if smoke else '')),run/'prediction_vs_reference.png')
            save_json(run/'evaluation_notes.json',dict(dataset='internal_validation',mape_definition=MAPE_RULE))
        save_json(run/'manifest.json',meta)
        if _refit_selection: emit('completed',artifact='final_model',path=str(run),metrics=dict(trees=best,training_rows=len(x['train'])),smoke=smoke)
        else: emit('completed',path=str(run),validation=meta['internal_validation_metrics'],smoke=smoke)
    except BaseException as exc:
        meta.update(status='cancelled' if isinstance(exc,(Cancelled,KeyboardInterrupt)) else 'failed',error=str(exc)); save_json(run/'manifest.json',meta); raise
    return run
