from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from . import expr as e
from .optimizer import _eval


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


@dataclass(frozen=True)
class CompiledUnit:
    _compiled_exprs: dict[str, Callable]
    _params: jax.Array
    _obs_dict: dict[str, float]

    def evaluate(self, exprName: str):
        return self._compiled_exprs[exprName](self._params, self._obs_dict)

    def observe(self, obs_dict) -> CompiledUnit:
        return CompiledUnit(self._compiled_exprs, self._params, obs_dict)


class Unit:
    """A unit is family of `Exprs` that share parameters and observations."""

    param_map = ParamMap()
    observations: frozenset[str]

    _exprs: dict[str, e.Expr] = dict()
    lossExpr: e.Expr | None = None

    def __init__(self, obs_names: frozenset[str]):
        self.observations = obs_names

    def register(self, name: str, expr: e.Expr) -> None:
        obs = {}
        _find_params(expr, self.param_map, obs)  # TODO: Make this do the check?
        for k in obs:
            if k not in self.observations:
                raise ValueError(f"Observation {k} not in {self.observations}")
        self._exprs[name] = expr

    def registerLoss(self, expr: e.Expr) -> None:
        self.lossExpr = expr

    def _compile(self, expr, params, obs_dict):
        def f(p, o):
            return _eval(expr, p, self.param_map, o)

        return jax.jit(f).lower(params, obs_dict).compile()

    def compile(self) -> CompiledUnit:
        params = jnp.full(self.param_map.count, 0.0)
        obs = {k: 0.0 for k in self.observations}
        compiled_exprs = {
            n: self._compile(e, params, obs) for n, e in self._exprs.items()
        }
        return CompiledUnit(compiled_exprs, params, obs)


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
