"""Persistent local reader. Request ids let the UI discard stale slider results."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
from collections import OrderedDict

import numpy as np

from .jobs import emit
from .volume_reader import VolumeReader
from .horizons import import_horizons, HorizonMap, overlay_line
from .well_spatial import prepare_wells, WellMap, well_overlay


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(); parser.add_argument("--index", required=True); args = parser.parse_args()
    index = Path(args.index).resolve(); cache_dir = index.parent / "display_cache"; cache_dir.mkdir(exist_ok=True)
    cache = cache_dir / f"slice_{os.getpid()}.npz"; partial = cache.with_name(cache.stem + ".partial.npz")
    try:
        with VolumeReader(index) as reader:
            layer_cache=OrderedDict()
            def mapped_horizon(filename):
                path=Path(filename).resolve();stamp=(path.stat().st_size,path.stat().st_mtime_ns)
                cached=layer_cache.get(path)
                if cached is None or cached[0]!=stamp:
                    cached=(stamp,HorizonMap.load(path,reader.index));layer_cache[path]=cached
                layer_cache.move_to_end(path)
                while len(layer_cache)>1 and sum(item[1].times_ms.size for item in layer_cache.values())>8_000_000:layer_cache.popitem(last=False)
                return cached[1]
            well_path = None; well_stamp = None; well_mapping = None
            emit("ready")
            for line in sys.stdin:
                request = {}; request_id = ""
                try:
                    request = json.loads(line); request_id = request.get("id", "")
                    if request.get("command") == "close": break
                    if not isinstance(request_id, str) or not re.fullmatch(r"[a-f0-9]{32}", request_id): raise ValueError("切片请求编号无效")
                    if request.get("command") == "import_horizon":
                        summary = import_horizons(reader.index, request["config"])
                        emit("horizon_loaded", id=request_id, **summary); continue
                    if request.get('command')=='interpolate_horizon':
                        from .horizon_interpolation import interpolate_horizon
                        summary=interpolate_horizon(reader.index,request['horizon_path'],request['horizon_name'],new_name=request['new_name'],max_span=request['max_span'],coordinates=request.get('coordinates','lines'))
                        emit('horizon_interpolated',id=request_id,**summary);continue
                    if request.get("command") == "import_wells":
                        from .volume_reader import require_fresh
                        require_fresh(reader.index.path)
                        summary = prepare_wells(reader, request["config"])
                        require_fresh(reader.index.path)
                        emit("wells_loaded", id=request_id, **summary); continue
                    mode = request["mode"]
                    times = None; name = request.get("horizon_name", "")
                    if request.get("horizon_path"):
                        horizons=mapped_horizon(request['horizon_path'])
                        times = horizons.layer(name)
                    if mode == "horizon":
                        if times is None: raise ValueError("请先导入并选择层位")
                        section = reader.horizon_section(times, name, request.get("sampling", "linear"),point_types=horizons.point_types[horizons.names.index(name)])
                    else: section = reader.section(mode, int(request["value"]))
                    overlay = overlay_line(reader.index, times, section, mode, request["value"]) if times is not None and mode in ("inline", "crossline") else np.array([])
                    layers=request.get('overlay_layers',[])
                    if len(layers)>64:raise ValueError('同时显示的层位最多 64 层')
                    layer_lines=[];layer_ids=[]
                    if mode in ('inline','crossline'):
                        if len(layers)*len(section.x)>4_000_000:raise ValueError('层位显示规模过大，请减少勾选层位')
                        for layer in layers:
                            layer_lines.append(overlay_line(reader.index,mapped_horizon(layer['path']).layer(layer['name']),section,mode,request['value']))
                            layer_ids.append(layer['id'])
                    wells = {}
                    if request.get("well_path") and mode in ('inline','crossline'):
                        path = Path(request['well_path']).resolve(); signature = (path.stat().st_size,path.stat().st_mtime_ns)
                        if path != well_path or signature != well_stamp:
                            well_mapping = WellMap.load(path,reader.index); well_path,well_stamp = path,signature
                        wells = {'well_'+k:v for k,v in well_overlay(well_mapping, reader.index, mode, request['value'], request['well_name'], request['well_curve']).items()}
                    np.savez_compressed(partial, values=section.values, x=section.x, y=section.y, xlabel=np.array(section.xlabel),
                                        ylabel=np.array(section.ylabel), title=np.array(section.title), invert_y=np.array(section.invert_y),
                                        limits=np.array(reader.index.limits), x_stride=np.array(reader.last_request["x_stride"]), y_stride=np.array(reader.last_request["y_stride"]),
                                        horizon_time=overlay, horizon_name=np.array(name),horizon_ids=np.array(layer_ids,dtype=str),
                                        horizon_point_types=reader.last_point_types if mode=='horizon' else np.empty((0,0),dtype=np.uint8),
                                        horizon_times=np.stack(layer_lines) if layer_lines else np.empty((0,len(section.x))), **wells)
                    partial.replace(cache); emit("slice", id=request_id, path=str(cache), stats=reader.last_request)
                except Exception as exc: emit("error", id=request_id, message=str(exc))
    except Exception as exc:
        emit("error", id="", message=str(exc)); return 1
    finally:
        for path in (cache, partial):
            if path.exists(): path.unlink()
    return 0


if __name__ == "__main__": raise SystemExit(main())
