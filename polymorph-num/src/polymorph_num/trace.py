import os

from .expr import Param

# To make it easier to debug errors about "params not found in loss", you can
# set the environment variable POLYMORPH_TRACE_PARAM to a parameter ID, and we
# raise an exception when a Param with that ID is created.
TRACE_ENV_VAR = "POLYMORPH_TRACE_PARAM"

_trace_id = int(os.environ.get(TRACE_ENV_VAR, "-1"))


class ParamTraceback(Exception):
    def __init__(self, param: Param) -> None:
        super().__init__(repr(param))
        self.add_note(f"See the above traceback for information about {param}.")
        self.add_note(f"NOTE: to stop seeing this traceback, unset {TRACE_ENV_VAR}.")


def maybe_trace_param(param: Param):
    if param.id == _trace_id:
        raise ParamTraceback(param)
    return param


def get_param_tracing_note(params: set[Param]):
    param = min(params, key=lambda x: x.id)
    return f"NOTE: try running with {TRACE_ENV_VAR}={param.id} for more information."
