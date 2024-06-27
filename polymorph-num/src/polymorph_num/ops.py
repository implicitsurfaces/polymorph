from . import expr

counter = 0


def param():
    global counter
    counter += 1
    return expr.Param(counter)


def observation(name):
    return expr.Observation(name)


def vec(value: list[float]):
    return expr.Arr(value)


def sum(n: expr.Expr):
    if n.dim == 1:
        raise ValueError()

    return expr.Sum(n)


def min(a, b):
    return expr.Binary(a, b, expr.BinOp.Min)


def max(a, b):
    return expr.Binary(a, b, expr.BinOp.Max)


def debug(tag, orig):
    return expr.Debug(tag, orig)
