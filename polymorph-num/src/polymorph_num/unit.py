from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import optimistix

from . import expr as e
from .eval import _eval
from .types import ObsDict

_DEFAULT_PRNG_KEY = jax.random.PRNGKey(0)


class ParamMap:
    def __init__(self):
        self.count = 0
        self.dict = dict()

    def add(self, node):
        if node not in self.dict:
            self.dict[node] = self.count
            self.count += 1

    def get(self, node, values):
        return values[self.dict[node]]

    def nodes(self):
        return set(self.dict.keys())


def _make_bfgs():
    return optimistix.BFGS(rtol=1e-5, atol=1e-6)


@dataclass(frozen=True)
class CompiledUnit:
    loss_fn: Callable
    compiled_exprs: dict[str, Callable]
    params: jax.Array
    obs_dict: ObsDict

    _expr_dims: dict[str, int] = field(default_factory=dict)
    solver: optimistix.AbstractMinimiser = field(default_factory=_make_bfgs)

    def evaluate(self, exprName: str):
        ans = self.compiled_exprs[exprName](self.params, self.obs_dict)
        return ans.item() if self._expr_dims[exprName] == 1 else ans

    def observe(self, obs_dict: ObsDict | dict[str, float]) -> CompiledUnit:
        # Passing float observations (as opposed to a JAX scalar) causes
        # recompilation, so make sure everything is a JAX array.
        new_obs = {k: jnp.asarray(v) for k, v in obs_dict.items()}
        return replace(self, obs_dict=new_obs)

    def minimize(self, max_steps=1000):
        soln = optimistix.minimise(
            self.loss_fn,
            self.solver,
            self.params,
            self.obs_dict,
            max_steps=max_steps,
            throw=False,
        )
        return replace(self, params=soln.value)


class Unit:
    """A unit is family of `Exprs` that share parameters and observations."""

    param_map: ParamMap
    observations: frozenset[str]

    _exprs: dict[str, e.Expr]
    lossExpr: e.Expr = e.as_expr(0.0)

    def __init__(
        self, obs_names: Sequence[str] | set[str] | frozenset[str] = frozenset()
    ):
        self.param_map = ParamMap()
        self.observations = frozenset(obs_names)
        self._exprs = dict()

    def register(self, name: str, expr: e.Expr) -> Unit:
        params = ParamMap()
        _find_params(expr, params, self.observations)

        extra_params = params.nodes() - self.param_map.nodes()
        if len(extra_params) > 0:
            raise ValueError(f"Expr has params not found in loss: {extra_params}")

        self._exprs[name] = expr
        return self

    def registerLoss(self, expr: e.Expr) -> Unit:
        _find_params(expr, self.param_map, self.observations)
        self.lossExpr = expr
        return self

    def _compile(self, expr, params, obs_dict: ObsDict):
        def eval_expr(p, d: ObsDict):
            return _eval(expr, p, self.param_map, d)

        return jax.jit(eval_expr).lower(params, obs_dict).compile()

    def compile(self, prng_key=_DEFAULT_PRNG_KEY) -> CompiledUnit:
        params = jax.random.uniform(prng_key, (self.param_map.count,))
        obs = {k: jnp.array(0.0) for k in self.observations}
        compiled_exprs = {
            n: self._compile(e, params, obs) for n, e in self._exprs.items()
        }
        dims = {n: e.dim for n, e in self._exprs.items()}

        def eval_loss(p, d: ObsDict):
            return _eval(self.lossExpr, p, self.param_map, d)

        loss_fn = jax.jit(eval_loss)
        return CompiledUnit(loss_fn, compiled_exprs, params, obs, dims)


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
