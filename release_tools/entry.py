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
        if module not in ("overpressure", "overpressure.slice_worker", "overpressure.seismic_geometry", "overpressure.browser_worker", "overpressure.rock_physics.worker"):
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
                from PySide6.QtCore import QEventLoop,QRect
                def wait_for(predicate):
                    loop=QEventLoop();poll=QTimer();timeout=QTimer();timeout.setSingleShot(True)
                    poll.timeout.connect(lambda:loop.quit() if predicate() else None)
                    timeout.timeout.connect(loop.quit);poll.start(20);timeout.start(45000);loop.exec()
                    poll.stop();timeout.stop();assert predicate(),'Frozen GUI operation timed out'
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
                import pandas as pd
                from overpressure.well_library import inspect_curve_file
                source=report.parent/'rp_synthetic.csv'
                pd.DataFrame(dict(Depth=np.linspace(100,200,20),POR=np.linspace(.08,.18,20),VP=np.linspace(4300,4000,20))).to_csv(source,index=False)
                window.library.wells=inspect_curve_file(source,forced_well='RP_synthetic',definitions=[dict(name='POR',column='POR',unit='fraction'),dict(name='VP',column='VP',unit='m/s')]);window.library.refresh()
                rp=window.rock_physics.workflow
                window.shell.workspace.module_dialogs[8].screen_guard.available_rect=lambda:QRect(0,0,1600,1050)
                window.shell.show_page(8);wait_for(lambda:rp.live.result.result is not None)
                assert rp.live.result.result['summary']['modeled_rows']==20
                result['rock_physics_live']=True
                window.shell.workspace.module_dialogs[8].grab().save(str(report.with_name(report.stem+'_rock_physics.png')))
                rp.start();wait_for(lambda:window.rock_physics.process is None and bool(window.shell.workspace.rock_physics_documents))
                result['rock_physics_save']=True
                window.shell.workspace.module_dialogs[8].hide()
                if len(sys.argv)>3:
                    from PySide6.QtCore import QEventLoop,QRect
                    from overpressure.seismic_import import SeismicImportDialog,STANDARD
                    def wait_for(predicate):
                        loop=QEventLoop();poll=QTimer();timeout=QTimer();timeout.setSingleShot(True)
                        poll.timeout.connect(lambda:loop.quit() if predicate() else None)
                        timeout.timeout.connect(loop.quit);poll.start(20);timeout.start(30000);loop.exec()
                        poll.stop();timeout.stop();assert predicate(),'SEG-Y GUI probe timed out'
                    source=Path(sys.argv[3]).resolve();before=source.stat()
                    dialog=SeismicImportDialog(window,source,dict(headers=STANDARD))
                    dialog.screen_guard.available_rect=lambda:QRect(0,0,1600,1050)
                    dialog.show();wait_for(lambda:dialog.preview_complete)
                    if not dialog.checked:
                        dialog.presets.setCurrentIndex(2);dialog.preview_button.click()
                        wait_for(lambda:dialog.preview_complete)
                    assert dialog.checked is not None,dialog.summary.toPlainText()
                    dialog.resize(800,900);app.processEvents()
                    dialog.grab().save(str(report.with_name(report.stem+'_segy_import.png')))
                    settings=dialog.values();dialog.accept()
                    window.shell.extra_volumes.append(settings);window.shell.refresh_tree()
                    window.shell.open_map_source(str(source));panel=window.shell.workspace.map
                    wait_for(lambda:panel.metadata is not None)
                    panel.create_line('inline',position=panel.metadata['inline_range'][0])
                    wait_for(lambda:bool(window.shell.workspace.sections))
                    section=next(iter(window.shell.workspace.sections.values())).widget().section
                    assert np.isfinite(section.values).any()
                    window.screen_guard.available_rect=lambda:QRect(0,0,1600,1050)
                    window.screen_guard.fit();window.resize(1440,940);window.shell.workspace.tile();app.processEvents()
                    window.grab().save(str(report.with_name(report.stem+'_segy_map.png')))
                    after=source.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
                    result['segy_import']=dict(passed=True,headers=settings['headers'],traces=panel.metadata['traces'],
                        inline=panel.metadata['inline_range'],crossline=panel.metadata['crossline_range'],section=True)
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
