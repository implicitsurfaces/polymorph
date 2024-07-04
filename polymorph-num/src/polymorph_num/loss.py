from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import optimistix

from . import expr as e
from .optimizer import _eval
from .types import ObsDict


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

    def __init__(self, obs_names: Sequence[str] | set[str]):
        self.param_map = ParamMap()
        self.observations = frozenset(obs_names)
        self._exprs = dict()

    def register(self, name: str, expr: e.Expr) -> Unit:
        obs = {}
        _find_params(expr, self.param_map, obs)  # TODO: Make this do the check?
        for k in obs:
            if k not in self.observations:
                raise ValueError(f"Observation {k} not in {self.observations}")
        self._exprs[name] = expr
        return self

    def registerLoss(self, expr: e.Expr) -> Unit:
        self.lossExpr = expr
        return self

    def _compile(self, expr, params, obs_dict: ObsDict):
        def eval_expr(p, d: ObsDict):
            return _eval(expr, p, self.param_map, d)

        return jax.jit(eval_expr).lower(params, obs_dict).compile()

    def compile(self) -> CompiledUnit:
        params = jnp.full(self.param_map.count, 0.0)
        obs = {k: jnp.array(0.0) for k in self.observations}
        compiled_exprs = {
            n: self._compile(e, params, obs) for n, e in self._exprs.items()
        }
        dims = {n: e.dim for n, e in self._exprs.items()}

        def eval_loss(p, d: ObsDict):
            return _eval(self.lossExpr, p, self.param_map, d)

        loss_fn = jax.jit(eval_loss)
        return CompiledUnit(loss_fn, compiled_exprs, params, obs, dims)


class Loss:
    def __init__(self, loss):
        self.loss = loss
        self.params = ParamMap()
        self.observations = {}
        _find_params(loss, self.params, self.observations)
        self.nodes = []

    def register_output(self, node):
        node_params = ParamMap()
        _find_params(node, node_params, {})
        extra_params = node_params.nodes() - self.params.nodes()
        if len(extra_params) > 0:
            raise ValueError(
                f"Cannot register node that contains parameter(s) not in loss: {extra_params}"
            )
        self.nodes.append(node)


def _find_params(node, params, observations):
    match node:
        case e.Broadcast(orig, _dim):
            _find_params(orig, params, observations)

        case e.Unary(orig, _op):
            _find_params(orig, params, observations)

        case e.Binary(left, right, _op):
            _find_params(left, params, observations)
            _find_params(right, params, observations)

        case e.Param(_id):
            params.add(node)

        case e.Observation(name):
            observations[name] = 0.0

        case e.Sum(orig):
            _find_params(orig, params, observations)
