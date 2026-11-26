"""Local development runtime discovery, without changing Conda environments."""
import os
from pathlib import Path
import sys
import shutil


def application_root() -> Path:
    return Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]


def user_data_root() -> Path:
    if not getattr(sys, "frozen", False):
        return application_root()
    root = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "PressurePrediction"
    root.mkdir(parents=True, exist_ok=True)
    return root


def default_project() -> Path:
    target = user_data_root() / "project.yaml"
    if getattr(sys, "frozen", False) and not target.exists():
        template = Path(sys._MEIPASS) / "project.example.yaml"
        shutil.copyfile(template, target)
    return target


def worker_command(module: str, arguments=()) -> tuple[str, list[str]]:
    if module not in ("overpressure", "overpressure.slice_worker", "overpressure.seismic_geometry", "overpressure.browser_worker", "overpressure.rock_physics.worker"):
        raise ValueError(f"Unsupported worker: {module}")
    if getattr(sys, "frozen", False):
        executable = application_root() / "PressureWorker.exe"
        if not executable.is_file():
            raise FileNotFoundError(f"计算组件缺失，请完整解压运行包：{executable}")
        return str(executable), ["--pressure-worker", module, *map(str, arguments)]
    return worker_python(), ["-u", "-m", module, *map(str, arguments)]


def worker_python() -> str:
    override = os.environ.get("PRESSURE_WORKER_PYTHON")
    candidates = [Path(override)] if override else []
    candidates.extend((Path.home() / ".conda" / "envs" / "pytorch_env" / "python.exe", Path(sys.executable)))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError("未找到计算环境的 Python")
