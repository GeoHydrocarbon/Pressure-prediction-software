"""Read line extents without loading traces, models or time-depth settings."""
import json
from pathlib import Path
import sys
import numpy as np


def preview_segy(config):
    """Validate a user-selected header mapping without reading trace amplitudes."""
    import segyio
    from .seismic_browser import SeismicDataset
    path=Path(config['path']).resolve();result=dict(passed=False)
    try:
        before=path.stat()
        with segyio.open(str(path),'r',ignore_geometry=True) as source:
            inline=np.asarray(source.attributes(config['headers']['inline'])[:])
            crossline=np.asarray(source.attributes(config['headers']['crossline'])[:])
            if not len(inline):raise ValueError('文件没有地震道')
            result.update(traces=source.tracecount,samples=len(source.samples),
                          inline=[int(inline.min()),int(inline.max())],
                          crossline=[int(crossline.min()),int(crossline.max())],
                          duplicate_pairs=int(source.tracecount-len(np.unique(np.column_stack((inline,crossline)),axis=0))))
        dataset=SeismicDataset(config)
        try:result.update(dataset.metadata())
        finally:dataset.close()
        after=path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            raise ValueError('预检期间文件发生变化，请重新预检')
        result.update(passed=True,stamp=dict(size=after.st_size,mtime_ns=after.st_mtime_ns))
    except Exception as exc:result['error']=str(exc)
    return result


def read_geometry(config):
    import segyio
    result = {'files': {}}
    geometries = {}
    for kind, filename in config['files'].items():
        try:
            path = Path(filename); before = path.stat()
            with segyio.open(str(path), 'r', ignore_geometry=True) as source:
                if source.tracecount < 1: raise ValueError('文件没有地震道')
                inline = np.asarray(source.attributes(config['headers']['inline'])[:]).copy()
                crossline = np.asarray(source.attributes(config['headers']['crossline'])[:]).copy()
                item = dict(inline=[int(inline.min()), int(inline.max())], crossline=[int(crossline.min()), int(crossline.max())], traces=source.tracecount)
            after = path.stat()
            if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns): raise ValueError('读取期间文件发生变化，请刷新范围')
            geometries[kind] = (inline,crossline); result['files'][kind] = item
        except Exception as exc: result['files'][kind] = {'error': str(exc)}
    if 'density' in geometries and 'impedance' in geometries:
        result['geometry_matches'] = all(np.array_equal(a,b) for a,b in zip(geometries['density'],geometries['impedance']))
    return result


def main():
    sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    sys.stdin.reconfigure(encoding='utf-8')
    try:
        config = json.load(sys.stdin)
        result=preview_segy(config) if config.get('operation')=='preview' else read_geometry(config)
        print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
        return 0
    except Exception as exc:
        print(json.dumps({'error':str(exc)},ensure_ascii=False),flush=True); return 1


if __name__ == '__main__': raise SystemExit(main())
