import node

counter = 0
def param():
    global counter
    counter += 1
    return node.Param(counter)

def observation(name):
    return node.Observation(name)

def vec(value: list[float]):
    return node.Vector(value)

def sqrt(v):
    return node.Unary(node.as_node(v), node.UnOp.Sqrt)

def sum(n: node.Node):
    if(n.dim == 1):
        raise ValueError()

    return node.Sum(n)
