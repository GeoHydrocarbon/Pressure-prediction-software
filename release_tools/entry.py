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
        if module not in ("overpressure", "overpressure.slice_worker", "overpressure.seismic_geometry", "overpressure.browser_worker"):
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
            try:
                import numpy as np
                from overpressure import __version__
                from overpressure.curve_settings import CurveSettingsDialog
                from overpressure.legend_settings import LegendSettingsDialog
                assert not window.library.wells and not window.shell.results
                assert not window.bundle_path.text() and not window.seismic_bundle.text()
                assert not window.traditional.applied_trends
                assert 7 in window.shell.workspace.module_dialogs
                viewer=window.shell.curve_viewer
                z=np.array([1000.,1001.,1002.])
                data=[dict(name=name,well='Synthetic GUI',source='synthetic',unit=unit,z=z,x=np.array(values,dtype=float),points=points)
                      for name,unit,values,points in [('AC','us/ft',[90,91,92],False),
                          ('Eaton pressure','MPa',[20,21,22],False),('Measured','MPa',[20.1,21.1,22.1],True)]]
                viewer.render(data)
                dialog=CurveSettingsDialog(viewer,viewer.setting_rows());rows=dialog.values();dialog.deleteLater()
                for row in rows:row['track']=1 if row['name']=='AC' else 2
                viewer.apply_plot_settings(rows)
                labels={row['name']:row['name'] for row in rows}
                legend_dialog=LegendSettingsDialog(viewer,viewer.legend_rows(),labels)
                legends,labels=legend_dialog.values();legend_dialog.deleteLater()
                legends[1]['settings']['position']='above'
                viewer.apply_legend_settings(legends,labels)
                window.shell.show_page(0);app.processEvents()
                sub=window.shell.workspace.documents[0];sub.close();window.shell.show_page(0)
                assert viewer.canvas.isVisible() and len(viewer.base_axes)==2
                assert len(viewer.base_axes[1].collections)==1 and len(viewer.base_axes[1].lines)==1
                viewer.canvas.figure.savefig(report.with_suffix('.png'))
                result=dict(gui='passed',version=__version__,title=window.windowTitle(),blank_project=True,
                            traditional_module=True,mixed_tracks=True,curve_styles=True,legend_settings=True,reopen_plot=True)
                report.write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')
            except BaseException as exc:
                import traceback
                report.write_text(json.dumps(dict(gui='failed',error=str(exc),traceback=traceback.format_exc())),encoding='utf-8')
                app.exit(1)
                return
            finally:window.close()
            app.quit()
        QTimer.singleShot(1500, finish)
        return app.exec()
    from overpressure.gui import main as gui_main
    return gui_main()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    raise SystemExit(main())
