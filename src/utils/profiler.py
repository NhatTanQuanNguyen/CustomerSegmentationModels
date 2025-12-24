import time
import os
import psutil
from contextlib import contextmanager

_process = psutil.Process(os.getpid())

@contextmanager
def profile_block(interval=0.01):
    mem_before = _process.memory_info().rss
    peak = mem_before
    t0 = time.perf_counter()

    running = True

    def monitor():
        nonlocal peak
        while running:
            rss = _process.memory_info().rss
            peak = max(peak, rss)
            time.sleep(interval)

    import threading
    th = threading.Thread(target=monitor, daemon=True)
    th.start()

    yield lambda: {
        "train_time_sec": round(time.perf_counter() - t0, 4),
        "memory_mb": round((peak - mem_before) / (1024 ** 2), 3)
    }

    running = False
