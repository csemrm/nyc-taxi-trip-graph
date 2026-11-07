"""PerformanceAnalyzer: timings, graph-size counters, optional memory (tracemalloc)."""
import time
import tracemalloc


def measure(fn, *args, memory=False, **kwargs):
    """Run fn; return (result, elapsed_s, peak_bytes_or_None)."""
    if memory:
        tracemalloc.start()
    t0 = time.perf_counter()
    try:
        out = fn(*args, **kwargs)
    finally:
        dt = time.perf_counter() - t0
        peak = None
        if memory:
            peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
    return out, dt, peak
