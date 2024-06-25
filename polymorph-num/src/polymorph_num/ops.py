from . import expr

counter = 0


def param():
    global counter
    counter += 1
    return expr.Param(counter)


def observation(name):
    return expr.Observation(name)


def vec(value: list[float]):
    return expr.Vector(value)


def sqrt(v):
    return expr.Unary(expr.as_expr(v), expr.UnOp.Sqrt)


def sum(n: expr.Expr):
    if n.dim == 1:
        raise ValueError()

    return expr.Sum(n)


def sigmoid(x: expr.Expr):
    return expr.Unary(x, expr.UnOp.Sigmoid)


def smoothabs(x: expr.Expr):
    return expr.Unary(x, expr.UnOp.SmoothAbs)
