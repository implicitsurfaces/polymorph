import jax
import jax.nn
import optimistix
from jax import Array
from jax import numpy as jnp

from . import expr as e
from .vec import Vec2


class Optimizer:
    def __init__(self, l):
        self.solver = optimistix.BFGS(rtol=1e-5, atol=1e-6)
        self.loss = l
        self.compiled_nodes = {}
        self.initial = jnp.full(self.loss.params.count, 0.0)

        def err(p, d):
            return self._eval(self.loss.loss, p, d)

        self.loss_fn = jax.jit(err)
        for n in l.nodes:
            self.compiled_nodes[n] = self._compile(n)

    def optimize(self, obs_dict):
        soln = optimistix.minimise(
            self.loss_fn, self.solver, self.initial, obs_dict, max_steps=1000, throw=False
        )
        self.initial = soln.value
        return Solution(self.compiled_nodes, soln.value, obs_dict)

    def _compile(self, node):
        def err(p, d):
            return self._eval(node, p, d)

        params = jnp.full(self.loss.params.count, 0.0)
        fn = jax.jit(err).lower(params, self.loss.observations).compile()
        return fn

    def _eval(self, node, p, d):
        print("tracing")
        return _eval(node, p, self.loss.params, d)


class Solution:
    def __init__(self, compiled_nodes, params, obs_dict):
        self.compiled_nodes = compiled_nodes
        self.params = params
        self.obs_dict = obs_dict

    def eval(self, node):
        return self.compiled_nodes[node](self.params, self.obs_dict)


def _eval(expr: e.Expr, params, param_map, obs_dict) -> Array:
    match expr:
        case e.Scalar(value):
            return jnp.array(value)

        case e.Arr(value):
            return jnp.array(value)

        case e.Broadcast(orig, _dim):
            return _eval(orig, params, param_map, obs_dict)

        case e.Unary(orig, op):
            o = _eval(orig, params, param_map, obs_dict)
            match op:
                case e.UnOp.Sqrt:
                    return jnp.sqrt(o)
                case e.UnOp.Sigmoid:
                    return jax.nn.sigmoid(o)
                case e.UnOp.SmoothAbs:
                    return o * jnp.tanh(10.0 * o)
                case e.UnOp.SoftPlus:
                    return jax.nn.softplus(50 * o) / 50

        case e.Binary(left, right, op):
            l = _eval(left, params, param_map, obs_dict)
            r = _eval(right, params, param_map, obs_dict)
            match op:
                case e.BinOp.Add:
                    return l + r
                case e.BinOp.Sub:
                    return l - r
                case e.BinOp.Mul:
                    return l * r
                case e.BinOp.Div:
                    return l / r
                case e.BinOp.Exp:
                    return l**r
                case e.BinOp.Min:
                    return jnp.minimum(l, r)
                case e.BinOp.Max:
                    return jnp.maximum(l, r)

        case e.Param():
            return param_map.get(expr, params)

        case e.Observation(name):
            return obs_dict[name]

        case e.Sum(orig):
            return jnp.sum(_eval(orig, params, param_map, obs_dict))

        case Vec2(x, y):
            return jnp.array(
                [
                    _eval(x, params, param_map, obs_dict),
                    _eval(y, params, param_map, obs_dict),
                ]
            )
        case e.Debug(tag, orig):
            o = _eval(orig, params, param_map, obs_dict)
            print(tag, o)
            return o

        case _:
            raise ValueError(f"Unknown expression: {expr}")
