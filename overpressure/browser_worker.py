"""Persistent SEG-Y browser protocol, kept outside the Qt interpreter."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

from .jobs import emit
from .seismic_browser import SeismicDataset


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stdin.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=True)
    args = parser.parse_args()
    folder = Path(args.cache).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    dataset = None
    emit('ready')
    try:
        for line in sys.stdin:
            request = {}
            try:
                request = json.loads(line)
                command = request['command']
                if command == 'close':
                    break
                if command == 'open':
                    if dataset is not None:
                        dataset.close()
                        dataset = None
                    dataset = SeismicDataset(request['config'])
                    target = folder / 'geometry.npz'
                    np.savez_compressed(target, **dataset.geometry())
                    wells, warnings = dataset.well_locations()
                    emit('opened', id=request['id'], path=str(target), metadata=dict(dataset.metadata(), wells=wells, warnings=warnings))
                    continue
                if dataset is None:
                    raise ValueError('请先载入地震数据')
                # Only one response file is live; the GUI acknowledges it by sending the next request.
                target = folder / 'display.npz'
                if command == 'map':
                    values, time_ms = dataset.map_slice(request['time_ms'])
                    np.savez_compressed(target, values=values)
                    emit('map', id=request['id'], path=str(target), time_ms=time_ms)
                elif command == 'section':
                    section, meta = dataset.section(request['mode'], coordinate_mode=request.get('coordinate_mode', 'lines'),
                                                    position=request.get('position'), vertices=request.get('vertices'))
                    np.savez_compressed(target, values=section.values, x=section.x, y=section.y,
                                        xlabel=np.array(section.xlabel), ylabel=np.array(section.ylabel), title=np.array(section.title))
                    emit('section', id=request['id'], view_id=request['view_id'], path=str(target), metadata=meta)
                else:
                    raise ValueError('未知浏览操作')
            except Exception as exc:
                emit('error', id=request.get('id', ''), message=str(exc))
    finally:
        if dataset is not None:
            dataset.close()
        for name in ('display.npz', 'geometry.npz'):
            (folder / name).unlink(missing_ok=True)
        try:
            folder.rmdir()
        except OSError:
            pass
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
