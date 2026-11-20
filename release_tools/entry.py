"""Shared entry point for the windowed application and console worker."""
import os
import runpy
import sys


def main():
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    if len(sys.argv) > 1 and sys.argv[1] == "--pressure-worker":
        module = sys.argv[2]
        if module not in ("overpressure", "overpressure.slice_worker", "overpressure.seismic_geometry"):
            raise ValueError("Unknown worker module")
        sys.argv = [module, *sys.argv[3:]]
        runpy.run_module(module, run_name="__main__")
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "--gui-smoke":
        import json
        from pathlib import Path
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication
        from overpressure.gui import MainWindow, configure_application
        report = Path(sys.argv[2])
        app = QApplication([])
        configure_application(app)
        window = MainWindow()
        window.show()
        def finish():
            report.write_text(json.dumps({"gui": "passed", "title": window.windowTitle()}, ensure_ascii=False), encoding="utf-8")
            window.close()
            app.quit()
        QTimer.singleShot(1500, finish)
        return app.exec()
    from overpressure.gui import main as gui_main
    return gui_main()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    raise SystemExit(main())
