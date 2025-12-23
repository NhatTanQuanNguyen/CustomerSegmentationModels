import time
import os
import psutil
from contextlib import contextmanager

_process = psutil.Process(os.getpid())

@contextmanager
def profile_block():
    """
    Đo:
    - Thời gian thực thi (seconds)
    - RAM tăng thêm (MB)
    """
    mem_before = _process.memory_info().rss
    t0 = time.perf_counter()

    yield lambda: {
        "train_time_sec": round(time.perf_counter() - t0, 4),
        "memory_mb": round(
            (_process.memory_info().rss - mem_before) / (1024 ** 2),
            3
        )
    }
