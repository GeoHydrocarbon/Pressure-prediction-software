"""Header-only maps and bounded sections for input SEG-Y and verified results."""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

from .seismic_view import Slice
from .volume_reader import MAX_DISPLAY_CELLS, sample_grid, scaled_coordinates, stamp


MAX_TRACES = 2_000_000
MAX_MAP_POINTS = 50_000


def path_samples(vertices, spacing, max_points=2048):
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not 2 <= len(points) <= 256 or not np.isfinite(points).all():
        raise ValueError('剖面线需要 2～256 个有效坐标点')
    points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-10]]
    if len(points) < 2:
        raise ValueError('剖面起点与终点不能相同')
    lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    count = min(max_points, max(2, int(np.ceil(lengths[-1] / max(spacing, 1e-9))) + 1))
    distance = np.linspace(0, lengths[-1], count)
    query = np.column_stack([np.interp(distance, lengths, points[:, axis]) for axis in (0, 1)])
    return distance, query


class SeismicDataset:
    """One open file per worker; cubes are never copied into the GUI process."""
    def __init__(self, config):
        import segyio
        self.config = dict(config)
        self.result_reader = None
        self.source = Path(config['path']).resolve()
        self.label = config.get('label') or self.source.stem
        self.unit = config.get('unit', '')
        self.result = self.source.is_dir()
        self.handle = None
        self._trees = {}
        try:
            if self.result:
                from .volume_reader import prepare_index, VolumeReader
                self.result_reader = VolumeReader(prepare_index(self.source))
                idx = self.result_reader.index
                self.handle = self.result_reader.pressure
                self.inline, self.crossline = idx.inline, idx.crossline
                self.offsets, self.ns, self.nt = idx.offsets, idx.ns, idx.nt
                self.interval_ms, self.t0_ms = idx.interval_ms, idx.t0_ms
                self.x, self.y = idx.x, idx.y
                coordinate_units = idx.coordinate_units
                self.unit = 'MPa'
                self.config['time_convention'] = idx.time_convention
            else:
                self.handle = segyio.open(str(self.source), 'r', ignore_geometry=True)
                f = self.handle
                if not 1 <= f.tracecount <= MAX_TRACES or len(f.samples) < 2:
                    raise ValueError('浏览支持 1～200 万道，每道至少两个采样点')
                fields = config['headers']
                self.inline = np.asarray(f.attributes(fields['inline'])[:]).astype(np.int64)
                self.crossline = np.asarray(f.attributes(fields['crossline'])[:]).astype(np.int64)
                if len(np.unique(np.column_stack((self.inline, self.crossline)), axis=0)) != f.tracecount:
                    raise ValueError('Inline / Crossline 存在重复位置，请在地震属性中核对道头')
                self.ns = len(f.samples)
                interval_us = int(f.bin[segyio.BinField.Interval])
                if interval_us <= 0 or np.any(np.asarray(f.attributes(117)[:]) != interval_us) or np.any(np.asarray(f.attributes(115)[:]) != self.ns):
                    raise ValueError('道采样间隔或采样点数不统一，请先整理 SEG-Y')
                self.interval_ms = interval_us / 1000.
                delay = np.asarray(f.attributes(109)[:], dtype=float)
                self.t0_ms = float(delay.min())
                offsets = (delay - self.t0_ms) / self.interval_ms
                if not np.allclose(offsets, np.round(offsets), atol=1e-7, rtol=0):
                    raise ValueError('各道起始时间不能对齐共同时间网格')
                self.offsets = np.round(offsets).astype(np.int64)
                self.nt = int(self.offsets.max()) + self.ns
                scalar = np.asarray(f.attributes(fields.get('coordinate_scalar', 71))[:])
                self.x = scaled_coordinates(f.attributes(fields.get('x', 181))[:], scalar)
                self.y = scaled_coordinates(f.attributes(fields.get('y', 185))[:], scalar)
                coordinate_units = np.asarray(f.attributes(fields.get('coordinate_units', 89))[:])
            if self.nt > 1_000_000:
                raise ValueError('共同时间轴超过 100 万个采样点，请先整理起始时间')
            self.xy_available = (self.x is not None and self.y is not None and
                                 np.isfinite(self.x).all() and np.isfinite(self.y).all() and
                                 (np.ptp(self.x) > 0 or np.ptp(self.y) > 0) and
                                 (coordinate_units is None or np.isin(coordinate_units, [0, 1]).all()))
            self.file_path = self.source / 'pressure.sgy' if self.result else self.source
            self.signature = stamp(self.file_path)
            self.map_indices = np.linspace(0, len(self.inline) - 1, min(MAX_MAP_POINTS, len(self.inline)), dtype=int)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.result_reader is not None:
            self.result_reader.close()
            self.result_reader = None
        elif self.handle is not None:
            self.handle.close()
        self.handle = None

    def check_fresh(self):
        if stamp(self.file_path) != self.signature:
            raise ValueError('地震文件发生变化，请重新载入底图')
        if self.result:
            from .volume_reader import require_fresh
            require_fresh(self.result_reader.index.path)

    def coordinates(self, mode):
        if mode == 'xy':
            if not self.xy_available:
                raise ValueError('没有可用的平面 XY 坐标，请使用线号底图')
            return np.column_stack((self.x, self.y))
        if mode != 'lines':
            raise ValueError('底图坐标必须为 xy 或 lines')
        return np.column_stack((self.crossline, self.inline)).astype(float)

    def tree(self, mode):
        if mode not in self._trees:
            tree = cKDTree(self.coordinates(mode))
            selected = self.map_indices[:5000]
            nearest = tree.query(self.coordinates(mode)[selected], k=min(2, len(self.inline)))[0]
            steps = nearest[:, 1] if nearest.ndim == 2 else np.array([1.])
            steps = steps[np.isfinite(steps) & (steps > 0)]
            self._trees[mode] = tree, float(np.median(steps)) if len(steps) else 1.
        return self._trees[mode]

    def metadata(self):
        return dict(path=str(self.source), label=self.label, unit=self.unit, result=self.result,
                    traces=len(self.inline), samples=self.ns, xy_available=bool(self.xy_available),
                    inline_range=[int(self.inline.min()), int(self.inline.max())],
                    crossline_range=[int(self.crossline.min()), int(self.crossline.max())],
                    time_range=[self.t0_ms, self.t0_ms + (self.nt - 1) * self.interval_ms],
                    interval_ms=self.interval_ms, map_points=len(self.map_indices),
                    time_convention=self.config.get('time_convention', 'twt'))

    def well_locations(self):
        """Resolve project wellheads on the same geometry used by the map."""
        from .well_spatial import read_asset
        locations = {}
        warnings = []
        for asset in self.config.get('well_assets', []):
            if asset.get('kind') != 'coordinates':
                continue
            try:
                _, frame = read_asset(asset)
                for name, group in frame.groupby('well'):
                    xy = group[['x', 'y']].to_numpy(dtype=float)
                    if not np.isfinite(xy).all() or not np.allclose(xy, xy[0], atol=1e-6, rtol=0):
                        warnings.append(f'{name} 的井口坐标不唯一，未显示')
                        continue
                    item = dict(name=str(name), x=float(xy[0, 0]), y=float(xy[0, 1]))
                    if self.xy_available:
                        distance, index = self.tree('xy')[0].query(xy[0])
                        if distance <= float(self.config.get('well_max_distance', 25.)):
                            item.update(inline=int(self.inline[index]), crossline=int(self.crossline[index]))
                    if name in locations and locations[name] != item:
                        warnings.append(f'{name} 的不同文件包含冲突井口坐标，未显示')
                        locations[name] = None
                    elif name not in locations:
                        locations[name] = item
            except Exception as exc:
                warnings.append('井位读取失败：' + str(exc))
        return [item for item in locations.values() if item is not None], warnings

    def geometry(self):
        rows = self.map_indices
        return dict(inline=self.inline[rows], crossline=self.crossline[rows],
                    x=self.x[rows] if self.xy_available else np.array([]),
                    y=self.y[rows] if self.xy_available else np.array([]))

    def values(self, row, sample):
        if self.result_reader:
            return self.result_reader.values(row, sample)
        data = np.asarray(self.handle.trace[int(row), sample], dtype=np.float32).reshape(-1)
        valid=np.isfinite(data)
        if self.config.get('positive_only'):valid &= data>0
        return np.where(valid, data, np.nan)

    def map_slice(self, time_ms):
        self.check_fresh()
        if not np.isfinite(time_ms):
            raise ValueError('时间必须为有限数')
        sample = int(round((float(time_ms) - self.t0_ms) / self.interval_ms))
        if not 0 <= sample < self.nt:
            raise ValueError('时间超出地震范围')
        values = np.full(len(self.map_indices), np.nan, dtype=np.float32)
        if self.result_reader:
            self.result_reader.last_request = dict(sample_values_read=0, traces_read=0)
        for i, row in enumerate(self.map_indices):
            local = sample - self.offsets[row]
            if 0 <= local < self.ns:
                values[i] = self.values(row, int(local))[0]
        self.check_fresh()
        return values, self.t0_ms + sample * self.interval_ms

    def section(self, mode, *, coordinate_mode='lines', position=None, vertices=None):
        self.check_fresh()
        coords = self.coordinates(coordinate_mode)
        if mode in ('inline', 'crossline'):
            if position is None or not np.isfinite(position):
                raise ValueError('请选择剖面位置')
            fixed, variable = (self.inline, self.crossline) if mode == 'inline' else (self.crossline, self.inline)
            available = np.unique(fixed)
            value = int(available[np.argmin(np.abs(available - position))])
            selected = np.flatnonzero(fixed == value)
            selected = selected[np.argsort(variable[selected])]
            low, high = int(variable[selected].min()), int(variable[selected].max())
            if high - low > 1_000_000:
                raise ValueError('线号跨度过大，请核对道头或使用任意线')
            sx, sy = sample_grid(high - low + 1, self.nt, MAX_DISPLAY_CELLS)
            axis = np.arange(low, high + 1, sx)
            rows = np.full(len(axis), -1, dtype=int)
            aligned = (variable[selected] - low) % sx == 0
            rows[(variable[selected[aligned]] - low) // sx] = selected[aligned]
            line = coords[selected]
            # Simplify only the displayed location line; sampled rows keep actual geometry.
            line = line[np.linspace(0, len(line) - 1, min(256, len(line)), dtype=int)]
            xlabel = 'Crossline' if mode == 'inline' else 'Inline'
            title = f'{mode.title()} {value}'
            selection = dict(position=value)
        elif mode in ('arbitrary', 'polyline'):
            tree, spacing = self.tree(coordinate_mode)
            axis, query = path_samples(vertices, spacing)
            distance, nearest = tree.query(query)
            rows = np.where(distance <= spacing * .8, nearest, -1)
            # Keep holes in the acquisition footprint as gaps, never extrapolate across them.
            sy = max(1, int(np.ceil(self.nt * len(rows) / MAX_DISPLAY_CELLS)))
            line = np.asarray(vertices, dtype=float)
            xlabel = 'Distance (map coordinate units)' if coordinate_mode == 'xy' else 'Distance (line grid units)'
            title = 'Arbitrary section' if mode == 'arbitrary' else 'Polyline section'
            selection = dict(vertices=line.tolist())
        else:
            raise ValueError('未知剖面方式')
        samples = np.arange(0, self.nt, sy)
        data = np.full((len(samples), len(rows)), np.nan, dtype=np.float32)
        if self.result_reader:
            self.result_reader.last_request = dict(sample_values_read=0, traces_read=0)
        for row in np.unique(rows[rows >= 0]):
            local = samples - self.offsets[row]
            good = (local >= 0) & (local < self.ns)
            if good.any():
                trace = self.values(row, slice(int(local[good][0]), int(local[good][-1]) + 1, sy))
                data[np.ix_(good, rows == row)] = trace[:, None]
        self.check_fresh()
        finite = data[np.isfinite(data)]
        limits = np.quantile(finite, [.01, .99]).tolist() if len(finite) else [0., 1.]
        section = Slice(data, axis, self.t0_ms + samples * self.interval_ms,
                        xlabel, 'Time (ms)', self.label + ' · ' + title, True)
        return section, dict(limits=limits, unit=self.unit, line=line.tolist(), mode=mode,
                             coordinate_mode=coordinate_mode, time_stride=sy,
                             missing_columns=int((rows < 0).sum()), source=str(self.source),positive_only=bool(self.config.get('positive_only')), **selection)
