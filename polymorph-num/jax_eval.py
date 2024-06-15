import node as n
import jax
from jax import Array
from jax import numpy as jnp

def eval(node: n.Node, params) -> Array:
    match node:
        case n.Scalar(value):
            return jnp.array(value)
        
        case n.Vector(value):
            return jnp.array(value)
        
        case n.Broadcast(orig, dim):
            return eval(orig, params)
        
        case n.Unary(orig, op):
            o = eval(orig, params)
            match op:
                case n.UnOp.Sqrt:
                    return jnp.sqrt(o)
                
        case n.Binary(left, right, op):
            l = eval(left, params)
            r = eval(right, params)
            match op:
                case n.BinOp.Add:
                    return l+r
                case n.BinOp.Sub:
                    return l-r
                case n.BinOp.Mul:
                    return l*r
                case n.BinOp.Div:
                    return l/r
                
        case n.Var(pos):
            return params[pos]
        
        case n.Sum(orig):
            return jnp.sum(eval(orig, params))
        
    raise ValueError()