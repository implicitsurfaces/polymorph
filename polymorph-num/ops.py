import node

def var(pos: int):
    return node.Var(pos)

def sqrt(v):
    return node.Unary(node.as_node(v), node.UnOp.Sqrt)