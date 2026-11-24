"""Normal-compaction calibration and independent, full-resolution Eaton prediction."""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .config import unit_factor
from .data import read_table, sha256_file
from .evaluation import evaluation_table
from .jobs import Cancelled, check_cancel, emit, save_json
from .well_points import match_pressure_points


def defaults(settings):
    cfg = dict(analysis_well='', analysis_curve='AC', ac_curve='AC', analysis_min=600., analysis_max=3800.,
               points=[[1000., 4.5], [3000., 3.8]], wells=[], applied_trends={},
               eaton_exponent=1.5, rock_density=2.45, fluid_density=1.03, gravity=9.8,
               depth_min=0., depth_max=10000., coefficient_min=.8, coefficient_max=2.,
               lithology_filter='all', lithology_column='岩性', max_gap_m=0., reference_curve='Pp')
    cfg.update(deepcopy(settings.get('traditional_prediction', {})))
    return cfg


def two_point_trend(points):
    points = np.asarray(points, dtype=float)
    if points.shape != (2, 2) or not np.isfinite(points).all() or (points[:, 0] < 0).any():
        raise ValueError('请输入两个有效的“深度（m）、LN(AC)”坐标点')
    if abs(points[1, 0] - points[0, 0]) < 1e-9:
        raise ValueError('两点深度不能相同')
    a = float((points[1, 1] - points[0, 1]) / (points[1, 0] - points[0, 0]))
    b = float(points[0, 1] - a * points[0, 0])
    if not np.isfinite([a, b]).all():
        raise ValueError('趋势系数超出数值范围')
    return dict(a=a, b=b, points=points.tolist(), depth_unit='m', ac_unit='us/ft',
                equation='LN(AC [us/ft]) = a * Depth [m] + b')


def validate_prediction(cfg):
    for name in ('eaton_exponent', 'rock_density', 'fluid_density', 'gravity'):
        if not np.isfinite(float(cfg[name])) or float(cfg[name]) <= 0:
            raise ValueError(f'{name} 必须为有限正数')
    if cfg['rock_density'] <= cfg['fluid_density']:
        raise ValueError('平均岩石密度必须大于流体密度')
    for left, right in [('depth_min', 'depth_max'), ('coefficient_min', 'coefficient_max')]:
        if not np.isfinite([cfg[left], cfg[right]]).all() or cfg[left] >= cfg[right]:
            raise ValueError('深度或压力系数的下限必须小于上限')
    if cfg['depth_min'] < 0 or not np.isfinite(cfg['max_gap_m']) or cfg['max_gap_m'] < 0:
        raise ValueError('深度和实测点匹配间隔必须为非负数')
    if cfg['lithology_filter'] not in ('all', 'mudstone'):
        raise ValueError('请选择全部有效 AC 点或仅泥岩')
    if not isinstance(cfg['wells'], list) or not cfg['wells'] or len(set(cfg['wells'])) != len(cfg['wells']):
        raise ValueError('请至少选择一口不重复的预测井')
    for well in cfg['wells']:
        trend = cfg['applied_trends'].get(well, {})
        if trend.get('confirmed') is not True:
            raise ValueError(f'{well} 尚未采用正常压实趋势，请先分析并应用趋势参数')
        two_point_trend(trend['points'])


class WellInputs:
    """Read selected curve references without requiring MLP features or pressure labels."""
    def __init__(self, project):
        self.project = project
        self.wells = {well['name']: well for well in project.settings.get('project_data', {}).get('wells', [])}
        self.tables = {}
        self.sources = {}

    def curve(self, well, name, physical_field):
        if well not in self.wells:
            raise ValueError(f'井库中不存在井 {well}')
        ref = next((c for c in self.wells[well]['curves'] if c['name'] == name), None)
        if ref is None:
            raise ValueError(f'{well} 缺少曲线 {name}')
        path = self.project.resolve(ref['path'])
        key = str(path)
        if key not in self.tables:
            before = path.stat()
            self.tables[key] = read_table(path, dtype={ref['well_column']: 'string'} if ref.get('well_column') else None)
            self.sources[key] = dict(path=key, sha256=sha256_file(path))
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('读取期间井数据文件发生变化，请重新运行')
        raw = self.tables[key]
        rows = np.arange(len(raw)) + 2
        if ref.get('well_column'):
            mask = raw[ref['well_column']].astype('string').str.strip().eq(ref.get('well_value', well)).fillna(False).to_numpy()
            rows, raw = rows[mask], raw.loc[mask]
        depth = pd.to_numeric(raw[ref['depth_column']], errors='coerce').to_numpy(dtype=float, na_value=np.nan) * unit_factor('Depth', ref['depth_unit'])
        if not len(depth) or not np.isfinite(depth).all() or (depth < 0).any() or pd.Series(depth.round(6)).duplicated().any():
            raise ValueError(f'{well} / {name} 的深度无效、重复或没有记录')
        values = pd.to_numeric(raw[ref['column']], errors='coerce').to_numpy(dtype=float, na_value=np.nan) * unit_factor(physical_field, ref['unit'])
        frame = pd.DataFrame(dict(well=well, Depth=depth, value=values, source_file=key, source_file_row=rows))
        return frame, raw.reset_index(drop=True)

    def ac(self, well, curve, lithology_column='岩性'):
        frame, raw = self.curve(well, curve, 'AC')
        frame = frame.rename(columns={'value': 'AC'})
        frame['lithology'] = raw[lithology_column].fillna('').astype(str).str.strip().to_numpy() if lithology_column in raw else ''
        frame.attrs['has_lithology'] = lithology_column in raw
        return frame.sort_values('Depth', kind='stable').reset_index(drop=True)

    def observations(self, well, reference_curve):
        refs = self.wells[well]['curves']
        measured = [ref for ref in refs if ref.get('role') == 'measurement']
        frames = []
        for ref in measured:
            frame, _ = self.curve(well, ref['name'], 'Pp')
            frames.append(frame.rename(columns={'value': 'Pp'}))
        if not measured and reference_curve and any(ref['name'] == reference_curve for ref in refs):
            frame, _ = self.curve(well, reference_curve, 'Pp')
            frames.append(frame.rename(columns={'value': 'Pp'}))
        result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(dict(
            well=pd.Series(dtype='string'),Depth=pd.Series(dtype=float),Pp=pd.Series(dtype=float),
            source_file=pd.Series(dtype='string'),source_file_row=pd.Series(dtype='int64')))
        result = result.loc[result.Pp.notna()].copy()
        result['source_row'] = result.source_file_row
        return result


def calculate_eaton(frame, trend, cfg):
    """Keep every input row and every finite computed value; flags never clip pressure."""
    result = frame.copy()
    z, ac = result.Depth.to_numpy(dtype=float), result.AC.to_numpy(dtype=float)
    a, b = trend['a'], trend['b']
    result['P_h'] = cfg['fluid_density'] * cfg['gravity'] * z / 1000.
    result['P_o'] = cfg['rock_density'] * cfg['gravity'] * z / 1000.
    valid_input = np.isfinite(ac) & (ac > 0) & np.isfinite(z) & (z >= 0)
    in_range = (z >= cfg['depth_min']) & (z <= cfg['depth_max'])
    lithology = np.ones(len(frame), dtype=bool)
    if cfg['lithology_filter'] == 'mudstone':
        if not frame.attrs.get('has_lithology'):
            raise ValueError(f'仅泥岩模式需要 AC 源文件中的岩性列：{cfg["lithology_column"]}')
        lithology = result.lithology.str.endswith('泥岩', na=False).to_numpy()
    selected = valid_input & in_range & lithology
    normal, pressure = np.full(len(frame), np.nan), np.full(len(frame), np.nan)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        normal[np.isfinite(z)] = np.exp(a * z[np.isfinite(z)] + b)
        pressure[selected] = (result.P_o.to_numpy()[selected] -
                              (result.P_o.to_numpy()[selected] - result.P_h.to_numpy()[selected]) *
                              (normal[selected] / ac[selected]) ** cfg['eaton_exponent'])
    good = selected & np.isfinite(normal) & (normal > 0) & np.isfinite(pressure)
    pressure[~good] = np.nan
    normal[~np.isfinite(normal)] = np.nan
    result['AC_normal'] = normal
    result['Pp_eaton'] = pressure
    result['Pp_pred'] = pressure
    coefficient = np.full(len(frame), np.nan)
    np.divide(pressure, result.P_h.to_numpy(), out=coefficient, where=good & (z > 0))
    result['pressure_coefficient'] = coefficient
    result['prediction_valid'] = good
    status = np.full(len(frame), 'calculated', dtype=object)
    status[~in_range] = 'outside_depth_range'
    status[~lithology & in_range] = 'excluded_lithology'
    status[~valid_input] = 'invalid_ac_or_depth'
    status[selected & ~good] = 'numeric_failure'
    result['calculation_status'] = status
    flags = np.full(len(frame), '', dtype=object)
    tolerance = 1e-9 * np.maximum(1., np.abs(result.P_o.to_numpy()))
    for mask, name in [(good & (pressure < 0), 'negative_pressure'),
                       (good & (pressure < result.P_h.to_numpy() - tolerance), 'below_hydrostatic'),
                       (good & (pressure > result.P_o.to_numpy() + tolerance), 'above_overburden'),
                       (good & (z == 0), 'zero_hydrostatic_reference'),
                       (np.isfinite(coefficient) & ((coefficient < cfg['coefficient_min']) | (coefficient > cfg['coefficient_max'])), 'coefficient_outside_range')]:
        flags[mask] = [old + (';' if old else '') + name for old in flags[mask]]
    result['quality_flags'] = flags
    return result


def _run_folder(project, kind):
    folder = project.resolve(project.settings['output_dir']) / 'traditional' / kind / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def analyze_trend(project, cancel_file=None):
    check_cancel(cancel_file)
    cfg = defaults(project.settings)
    trend = two_point_trend(cfg['points'])
    if not np.isfinite([cfg['analysis_min'], cfg['analysis_max']]).all() or not 0 <= cfg['analysis_min'] < cfg['analysis_max']:
        raise ValueError('趋势分析深度范围必须满足 0 ≤ 起始深度 < 结束深度')
    inputs = WellInputs(project)
    frame = inputs.ac(cfg['analysis_well'], cfg['analysis_curve'])
    check_cancel(cancel_file)
    frame = frame.loc[frame.Depth.between(cfg['analysis_min'], cfg['analysis_max'])].copy()
    valid = np.isfinite(frame.AC) & (frame.AC > 0)
    if valid.sum() < 2 or frame.loc[valid, 'Depth'].nunique() < 2:
        raise ValueError('分析范围内至少需要两个不同深度的有效 AC 数据点')
    frame['ln_ac'] = np.nan
    frame.loc[valid, 'ln_ac'] = np.log(frame.loc[valid, 'AC'])
    out = _run_folder(project, 'trends')
    meta = dict(schema_version=1, status='running', artifact='traditional_trend', well=cfg['analysis_well'],
                ac_curve=cfg['analysis_curve'], analysis_min=cfg['analysis_min'], analysis_max=cfg['analysis_max'],
                trend=trend, sources=list(inputs.sources.values()), rows=len(frame), valid_points=int(valid.sum()))
    save_json(out / 'traditional.json', meta)
    try:
        check_cancel(cancel_file)
        frame.to_csv(out / 'trend_data.csv', index=False, encoding='utf-8-sig')
        from .traditional_figures import trend_figure
        from .charts import save_figure
        save_figure(trend_figure(frame, trend, cfg['analysis_well']), out / 'normal_compaction.png')
        check_cancel(cancel_file)
        meta.update(status='completed', data_sha256=sha256_file(out / 'trend_data.csv'))
        save_json(out / 'traditional.json', meta)
        emit('completed', artifact='traditional_trend', path=str(out), metrics=dict(a=trend['a'], b=trend['b'], valid_points=int(valid.sum())))
    except BaseException as exc:
        meta.update(status='cancelled' if isinstance(exc, (Cancelled, KeyboardInterrupt)) else 'failed', error=str(exc))
        save_json(out / 'traditional.json', meta)
        raise
    return out


def predict_eaton(project, cancel_file=None):
    check_cancel(cancel_file)
    cfg = defaults(project.settings)
    validate_prediction(cfg)
    out = _run_folder(project, 'predictions')
    inputs = WellInputs(project)
    meta = dict(schema_version=1, status='running', artifact='traditional_prediction', method='eaton', settings=cfg,
                pressure_unit='MPa', coefficient_definition='Pp_eaton / P_h',
                overburden_method='constant_mean_density', canonical_units=dict(Depth='m', AC='us/ft', density='g/cc'),
                formula='Pp = Po - (Po - Ph) * (AC_normal / AC) ** n; AC_normal = exp(a*z+b)',
                evaluation_scope='comparison with supplied pressure references; parameter calibration may use the same wells',
                per_well_files=[])
    save_json(out / 'traditional.json', meta)
    try:
        frames, points, descriptions = [], [], []
        folder = out / 'per_well'
        folder.mkdir()
        for number, name in enumerate(cfg['wells'], 1):
            check_cancel(cancel_file)
            emit('stage', message=f'Eaton 预测：{name}（{number}/{len(cfg["wells"])}）')
            frame = inputs.ac(name, cfg['ac_curve'], cfg['lithology_column'])
            adopted = cfg['applied_trends'][name]
            trend = two_point_trend(adopted['points'])
            result = calculate_eaton(frame, trend, cfg)
            if not result.prediction_valid.any():
                raise ValueError(f'{name} 在所选深度和岩性范围内没有有效 Eaton 预测点')
            observations = inputs.observations(name, cfg['reference_curve'])
            matched = match_pressure_points(result, observations, cfg['max_gap_m'])
            points.append(matched)
            result['Pp'] = np.nan
            filename = f'{number:03d}_' + (re.sub(r'[^\w.-]+', '_', name)[:80] or 'well') + '.csv'
            result.to_csv(folder / filename, index=False, encoding='utf-8-sig')
            descriptions.append(dict(well=name, path='per_well/' + filename, rows=len(result),
                                     depth_min=float(result.Depth.min()), depth_max=float(result.Depth.max()),
                                     valid_points=int(result.prediction_valid.sum()), flagged_points=int(result.quality_flags.ne('').sum()),
                                     trend=trend, adopted_from=adopted.get('source_well', ''), sha256=sha256_file(folder / filename)))
            frames.append(result)
            check_cancel(cancel_file)
            from .traditional_figures import eaton_figure
            from .charts import save_figure
            save_figure(eaton_figure(result, matched, name), out / f'eaton_{number:03d}.png')
            emit('traditional_progress', completed=number, total=len(cfg['wells']))
        combined = pd.concat(frames, ignore_index=True)
        combined_points = pd.concat(points, ignore_index=True)
        combined.to_csv(out / 'predictions.csv', index=False, encoding='utf-8-sig')
        combined_points.to_csv(out / 'reference_points.csv', index=False, encoding='utf-8-sig')
        table = evaluation_table(combined_points)
        table.to_csv(out / 'evaluation_metrics.csv', index=False, encoding='utf-8-sig')
        check_cancel(cancel_file)
        meta.update(status='completed', sources=list(inputs.sources.values()), per_well_files=descriptions,
                    data_sha256=sha256_file(out / 'predictions.csv'), reference_sha256=sha256_file(out / 'reference_points.csv'),
                    prediction_rows=len(combined), valid_points=int(combined.prediction_valid.sum()),
                    matched_points=int(combined_points.supervised_valid.sum()))
        save_json(out / 'traditional.json', meta)
        emit('completed', artifact='traditional_prediction', path=str(out),
             metrics=dict(wells=len(frames), valid_points=meta['valid_points'], matched_points=meta['matched_points']))
    except BaseException as exc:
        meta.update(status='cancelled' if isinstance(exc, (Cancelled, KeyboardInterrupt)) else 'failed', error=str(exc))
        save_json(out / 'traditional.json', meta)
        raise
    return out
