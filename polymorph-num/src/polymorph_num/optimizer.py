from . import node as n, loss
from . import point
import jax
from jax import Array
from jax import numpy as jnp
import jax.nn
import optimistix


class Optimizer:
    def __init__(self, l):
        self.solver = optimistix.BFGS(rtol=1e-5, atol=1e-6)
        self.loss = l
        self.compiled_nodes = {}

        def err(p, d):
            return self._eval(self.loss.loss, p, d)

        self.loss_fn = jax.jit(err)
        for n in l.nodes:
            self.compiled_nodes[n] = self._compile(n)

    def optimize(self, obs_dict):
        params = jnp.full(self.loss.params.count, 0.0)
        soln = optimistix.minimise(
            self.loss_fn, self.solver, params, obs_dict, max_steps=1000, throw=False
        )
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


def _eval(node: n.Node, params, param_map, obs_dict) -> Array:
    match node:
        case n.Scalar(value):
            return jnp.array(value)

        case n.Vector(value):
            return jnp.array(value)

        case n.Broadcast(orig, dim):
            return _eval(orig, params, param_map, obs_dict)

        case n.Unary(orig, op):
            o = _eval(orig, params, param_map, obs_dict)
            match op:
                case n.UnOp.Sqrt:
                    return jnp.sqrt(o)
                case n.UnOp.Sigmoid:
                    return jax.nn.sigmoid(o)
                case n.UnOp.SmoothAbs:
                    return o * jnp.tanh(10.0 * o)

        case n.Binary(left, right, op):
            l = _eval(left, params, param_map, obs_dict)
            r = _eval(right, params, param_map, obs_dict)
            match op:
                case n.BinOp.Add:
                    return l + r
                case n.BinOp.Sub:
                    return l - r
                case n.BinOp.Mul:
                    return l * r
                case n.BinOp.Div:
                    return l / r

        case n.Param():
            return param_map.get(node, params)

        case n.Observation(name):
            return obs_dict[name]

        case n.Sum(orig):
            return jnp.sum(_eval(orig, params, param_map, obs_dict))

        case point.Point(x, y):
            return jnp.array(
                [
                    _eval(x, params, param_map, obs_dict),
                    _eval(y, params, param_map, obs_dict),
                ]
            )
