"""Training orchestration; algorithm backends never choose the validation split."""
from ..algorithms.registry import get_algorithm


def train(project, *, smoke=False, cancel_file=None, _prepared=None, _refit_selection=None):
    cfg = project.settings
    algorithm = get_algorithm(cfg['algorithm']['name'])
    algorithm.require_available()
    if cfg['physics']['enabled'] and cfg['physics']['method'] not in algorithm.training_physics:
        raise ValueError(f'{algorithm.title} 尚不支持此训练物理约束')
    if _prepared is None and cfg['data'].get('split_method') == 'leave_one_well_out':
        return cross_validate(project, smoke=smoke, cancel_file=cancel_file)
    return algorithm.train(project, smoke=smoke, cancel_file=cancel_file,
                           _prepared=_prepared, _refit_selection=_refit_selection)


def cross_validate(project, *, smoke=False, cancel_file=None):
    algorithm = get_algorithm(project.settings['algorithm']['name'])
    algorithm.require_available()
    from ..cross_validation import cross_validate as run
    return run(project, smoke=smoke, cancel_file=cancel_file)


def refit(project, cv_folder, *, smoke=False, cancel_file=None):
    get_algorithm(project.settings['algorithm']['name']).require_available()
    from ..refit import refit as run
    return run(project, cv_folder, smoke=smoke, cancel_file=cancel_file)
