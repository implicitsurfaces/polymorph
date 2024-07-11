import functools
import logging
import time


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
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            end = time.perf_counter()
            logger.debug(f"{fn.__name__} took {((end - start) * 1000.):.3f}ms")
            return result

        return wrapper

    return dec
