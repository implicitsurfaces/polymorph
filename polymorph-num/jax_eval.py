import node as n
import params
import jax
from jax import Array
from jax import numpy as jnp
import optimistix

def minimize(loss: n.Node):
    map = params.find_params(loss)
    theta = jnp.full(map.count, 0.0)
    def err(p, _):
        return _eval(loss, p, map)
    solver = optimistix.BFGS(rtol=1e-5, atol=1e-6)
    solution = optimistix.minimise(err, solver, theta, max_steps=1000, throw=False)
    return Solution(solution.value, map)

class Solution:
    def __init__(self, theta, map):
        self.theta = theta
        self.map = map

    def eval(self, node: n.Node) -> Array:
        return _eval(node, self.theta, self.map)

def _eval(node: n.Node, theta, map) -> Array:
    match node:
        case n.Scalar(value):
            return jnp.array(value)
        
        case n.Vector(value):
            return jnp.array(value)
        
        case n.Broadcast(orig, dim):
            return _eval(orig, theta, map)
        
        case n.Unary(orig, op):
            o = _eval(orig, theta, map)
            match op:
                case n.UnOp.Sqrt:
                    return jnp.sqrt(o)
                
        case n.Binary(left, right, op):
            l = _eval(left, theta, map)
            r = _eval(right, theta, map)
            match op:
                case n.BinOp.Add:
                    return l+r
                case n.BinOp.Sub:
                    return l-r
                case n.BinOp.Mul:
                    return l*r
                case n.BinOp.Div:
                    return l/r
                
        case n.Param():
            return map.get(node, theta)
        
        case n.Sum(orig):
            return jnp.sum(_eval(orig, theta, map))
        
    raise ValueError()