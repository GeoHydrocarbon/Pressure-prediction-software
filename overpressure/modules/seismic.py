"""Shared three-dimensional prediction entry point for range and full-volume jobs."""


def predict(project, bundle_path, *, full=False, line_range=None, trace_start=0,
            trace_count=32, trace_stride=1, block_traces=None, cancel_file=None):
    from ..predict_volume import predict as run
    return run(project, bundle_path, full=full, line_range=line_range,
               trace_start=trace_start, trace_count=trace_count, trace_stride=trace_stride,
               block_traces=block_traces, cancel_file=cancel_file)
