import functools
import logging
import time
from contextlib import contextmanager


def log_perf(logger: logging.Logger):
    """
    A decorator that logs the time taken for a function to execute.
    Usage:

        @log_perf(render_log)
        def render_scene():
            ...

    And you'll get a log like this:

        DEBUG:render:render_scene took 14.035ms
    """

    def dec(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()

            result = fn(*args, **kwargs)
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.debug(f"{fn.__name__} took {duration_ms:.3f}ms")
            return result

        return wrapper

    return dec


@contextmanager
def perf_logging(logger: logging.Logger, desc: str):
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        logger.debug(f"{desc} took {duration_ms:.3f}ms")
