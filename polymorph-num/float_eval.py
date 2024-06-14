import node as n
import math

def eval(node: n.Node, params) -> float:
    match node:
        case n.Constant(value):
            return value
        
        case n.Unary(orig, op):
            o = eval(orig, params)
            match op:
                case n.UnOp.Sqrt:
                    return math.sqrt(o)
                
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
        
    raise ValueError()