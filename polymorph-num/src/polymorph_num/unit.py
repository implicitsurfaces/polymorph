from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import optax
import optax.tree_utils as otu

from polymorph_num.util import log_perf
from polymorph_num.vec import Vec2

from . import expr as e
from .eval import _eval
from .trace import get_param_tracing_note
from .types import ObsDict

_DEFAULT_PRNG_KEY = jax.random.PRNGKey(0)

compile_log = logging.getLogger("compile")
jaxpr_log = logging.getLogger("jaxpr")
lbfgs_log = logging.getLogger("lbfgs")


class ParamMap:
    def __init__(self):
        self.count = 0
        self.dict = dict()

    def __repr__(self):
        return f"ParamMap({self.dict})"

    def add(self, node):
        if node not in self.dict:
            self.dict[node] = self.count
            self.count += 1

    def get(self, node, values):
        return values[self.dict[node]]

    def nodes(self):
        return set(self.dict.keys())


_lbfgs = optax.lbfgs()


@partial(jax.jit, static_argnames=["fun"])
def _run_lbfgs(init_params, lbfgs_state, fun, max_steps, tolerance, **kwargs):
    value_and_grad_fun = optax.value_and_grad_from_state(fun)

    def step(carry):
        params, state = carry
        value, grad = value_and_grad_fun(params, **kwargs, state=state)

        updates, state = _lbfgs.update(
            grad, state, params, **kwargs, value=value, grad=grad, value_fn=fun
        )
        params = optax.apply_updates(params, updates)
        return params, state

    def continuing_criterion(carry):
        _, state = carry
        iter_num = otu.tree_get(state, "count")
        grad = otu.tree_get(state, "grad")
        err = otu.tree_l2_norm(grad)
        return (iter_num == 0) | ((iter_num < max_steps) & (err >= tolerance))

    if lbfgs_state is None:
        state = _lbfgs.init(init_params)
    else:
        # reset iteration count leftover from previous solve
        state = otu.tree_set(lbfgs_state, count=0)

    init_carry = (init_params, state)
    final_params, final_state = jax.lax.while_loop(
        continuing_criterion, step, init_carry
    )
    return final_params, final_state


def key_generator(start_key=_DEFAULT_PRNG_KEY):
    key = start_key
    while True:
        key, subkey = jax.random.split(key)
        yield subkey


def _return_val(val, dim: int) -> Any:
    if isinstance(val, tuple):
        return (_return_val(val[0], dim), _return_val(val[1], dim))
    return val.item() if dim == 1 else val


@dataclass(frozen=True)
class CompiledUnit:
    loss_fn: Callable
    compiled_exprs: dict[str, Callable]
    params: jax.Array
    obs_dict: ObsDict
    param_map: ParamMap = field(default_factory=ParamMap)
    _expr_dims: dict[str, int] = field(default_factory=dict)
    _minimizer_state: None | optax.OptState = None

    def run(self, expr: e.Expr | Vec2):
        fun = _compile_expr(expr, self.params, self.param_map, self.obs_dict)
        val = fun(self.params, self.obs_dict)
        return _return_val(val, expr.dim)

    def evaluate(self, exprName: str):
        ans = self.compiled_exprs[exprName](self.params, self.obs_dict)
        dim = self._expr_dims[exprName]

        return _return_val(ans, dim)

    def observe(self, obs_dict: ObsDict | dict[str, float]) -> CompiledUnit:
        # Passing float observations (as opposed to a JAX scalar) causes
        # recompilation, so make sure everything is a JAX array.
        new_obs = {k: jnp.asarray(v) for k, v in obs_dict.items()}
        if self.obs_dict == new_obs:
            return self
        else:
            # if obs have changed, we also need to reset the internal minimizer state.
            return replace(self, obs_dict=new_obs, _minimizer_state=None)

    @log_perf(lbfgs_log)
    def minimize(self, max_steps=1000):
        soln, state = _run_lbfgs(
            self.params,
            self._minimizer_state,
            self.loss_fn,
            max_steps,
            tolerance=1e-3,
            d=self.obs_dict,
        )
        iter_num = otu.tree_get(state, "count")
        lbfgs_log.debug(f"Minimization used {iter_num} of max {max_steps} steps")
        return replace(self, params=soln, _minimizer_state=state)

    def minimize_within_time_limit(self, time_limit_ms=10.0):
        start_time = time.time()
        single_step_result = self.minimize(max_steps=1)
        single_step_ms = (time.time() - start_time) * 1000
        remaining_time_ms = time_limit_ms - single_step_ms
        if remaining_time_ms < 0:
            lbfgs_log.error(
                f"Minimization time limit of {time_limit_ms} exceeded by {-remaining_time_ms}ms"
            )
            return single_step_result
        else:
            max_steps = math.floor(remaining_time_ms / single_step_ms)
            result = single_step_result.minimize(max_steps=max_steps)
            time_used_ms = (time.time() - start_time) * 1000
            if time_used_ms > time_limit_ms:
                lbfgs_log.error(
                    f"Minimization time limit of {time_limit_ms} exceeded by {time_used_ms - time_limit_ms}ms"
                )
            return result


def eval_expr(expr: e.Expr | Vec2, params_map, params, obs_dict):
    random_key = key_generator()
    if isinstance(expr, Vec2):
        return (
            _eval(expr.x, params, params_map, obs_dict, random_key, {}),
            _eval(expr.y, params, params_map, obs_dict, random_key, {}),
        )
    return _eval(expr, params, params_map, obs_dict, random_key, {})


def _compile_expr(expr, params, param_map, obs_dict):
    return (
        jax.jit(eval_expr, static_argnums=(0, 1))
        .lower(expr, param_map, params, obs_dict)
        .compile()
    )


def _make_jaxpr(expr, params, param_map, obs_dict):
    # This should always mirror the structure in `_compile_expr`.
    return (jax.make_jaxpr(eval_expr, static_argnums=(0, 1)))(
        expr, param_map, params, obs_dict
    )


class Unit:
    """A unit is family of `Exprs` that share parameters and observations."""

    param_map: ParamMap
    observations: frozenset[str]

    _exprs: dict[str, e.Expr | Vec2]
    lossExpr: e.Expr = e.ZERO

    def __init__(
        self, obs_names: Sequence[str] | set[str] | frozenset[str] = frozenset()
    ):
        self.param_map = ParamMap()
        self.observations = frozenset(obs_names)
        self._exprs = dict()

    def register(self, name: str, expr: e.Expr | Vec2) -> Unit:
        params = ParamMap()
        _find_params(expr, params, self.observations)

        extra_params = params.nodes() - self.param_map.nodes()
        if len(extra_params) > 0:
            err = ValueError(f"Expr has params not found in loss: {extra_params}")
            err.add_note(get_param_tracing_note(extra_params))
            raise err

        self._exprs[name] = expr
        return self

    def registerLoss(self, expr: e.Expr) -> Unit:
        _find_params(expr, self.param_map, self.observations)
        self.lossExpr = expr
        return self

    def compile(self, prng_key=_DEFAULT_PRNG_KEY) -> CompiledUnit:
        start_time = time.time()

        params = jax.random.uniform(prng_key, (self.param_map.count,))
        obs = {k: jnp.array(0.0) for k in self.observations}
        compiled_exprs = {}
        for name, expr in self._exprs.items():
            start_time = time.time()
            compiled_exprs[name] = _compile_expr(expr, params, self.param_map, obs)
            compile_log.debug(f"Compiled {name} in {time.time() - start_time:.2f}s")

            if jaxpr_log.isEnabledFor(logging.DEBUG):
                jaxpr = _make_jaxpr(expr, params, self.param_map, obs)
                jaxpr_log.debug(f"{name} ↓\n{jaxpr}")
        dims = {n: e.dim for n, e in self._exprs.items()}

        def eval_loss(p, d: ObsDict):
            return _eval(self.lossExpr, p, self.param_map, d, key_generator(), {})

        loss_fn = jax.jit(eval_loss)

        compile_log.debug(f"Unit compilation: {time.time() - start_time:.2f}s")

        return CompiledUnit(loss_fn, compiled_exprs, params, obs, self.param_map, dims)


def _find_params(expr, params: ParamMap, obs_names: frozenset[str]):
    match expr:
        case e.Broadcast(orig, _dim):
            _find_params(orig, params, obs_names)

        case e.Unary(orig, _op):
            _find_params(orig, params, obs_names)

        case e.Binary(left, right, _op):
            _find_params(left, params, obs_names)
            _find_params(right, params, obs_names)

        case e.Param(_id):
            params.add(expr)

        case e.Observation(name):
            if name not in obs_names:
                raise ValueError(f"Observation '{name}' not found in {obs_names}")

        case e.Sum(orig):
            _find_params(orig, params, obs_names)
