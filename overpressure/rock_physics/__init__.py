"""Rock-physics model catalog and validated numerical calculations.

Catalog access uses source metadata only. SciPy and the vendored numerical
modules are imported only when a calculation is requested.
"""
from .catalog import catalog


def calculate(model_id, parameters):
    from .calculation import calculate as run
    return run(model_id, parameters)


def save_result(result, output_dir):
    from .storage import save_result as save
    return save(result, output_dir)


__all__ = ['catalog', 'calculate', 'save_result']
