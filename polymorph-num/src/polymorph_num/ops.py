from . import expr
from .trace import maybe_trace_param

counter = 0


def param():
    global counter
    counter += 1
    return maybe_trace_param(expr.Param(counter))


def observation(name):
    return expr.Observation(name)


def vec(value: list[float]):
    return expr.Arr(value)


def sum(n: expr.Expr):
    if n.dim == 1:
        raise ValueError()

    return expr.Sum(n)


def mean(n: expr.Expr):
    if n.dim == 1:
        raise ValueError()

    return expr.Sum(n) / n.dim


def min(a: expr.Num, b: expr.Num):
    return expr.broadcast_binary(expr.as_expr(a), expr.as_expr(b), expr.BinOp.Min)


def max(a: expr.Num, b: expr.Num):
    return expr.broadcast_binary(expr.as_expr(a), expr.as_expr(b), expr.BinOp.Max)


def atan2(a: expr.Num, b: expr.Num):
    return expr.broadcast_binary(expr.as_expr(a), expr.as_expr(b), expr.BinOp.ArcTan2)


def if_gt(a: expr.Num, b: expr.Num, if_a_gt_b: expr.Num, if_a_le_b: expr.Num):
    args = expr.broacast_args(
        expr.as_expr(a),
        expr.as_expr(b),
        expr.as_expr(if_a_gt_b),
        expr.as_expr(if_a_le_b),
    )
    return expr.ComparisonIf(*args, op=expr.ComparisonOp.Gt)


def if_ge(a: expr.Num, b: expr.Num, if_a_ge_b: expr.Num, if_a_lt_b: expr.Num):
    args = expr.broacast_args(
        expr.as_expr(a),
        expr.as_expr(b),
        expr.as_expr(if_a_ge_b),
        expr.as_expr(if_a_lt_b),
    )
    return expr.ComparisonIf(*args, op=expr.ComparisonOp.Ge)


def if_lt(a: expr.Num, b: expr.Num, if_a_lt_b: expr.Num, if_a_ge_b: expr.Num):
    return if_ge(a, b, if_a_ge_b, if_a_lt_b)


def if_le(a: expr.Num, b: expr.Num, if_a_le_b: expr.Num, if_a_gt_b: expr.Num):
    return if_gt(a, b, if_a_gt_b, if_a_le_b)


def if_eq(a: expr.Num, b: expr.Num, if_a_eq_b: expr.Num, if_a_ne_b: expr.Num):
    args = expr.broacast_args(
        expr.as_expr(a),
        expr.as_expr(b),
        expr.as_expr(if_a_eq_b),
        expr.as_expr(if_a_ne_b),
    )
    return expr.ComparisonIf(*args, op=expr.ComparisonOp.Eq)


def if_ne(a: expr.Num, b: expr.Num, if_a_ne_b: expr.Num, if_a_eq_b: expr.Num):
    return if_eq(a, b, if_a_eq_b, if_a_ne_b)


def clamp(value: expr.Num, min_value: expr.Num, max_value: expr.Num):
    return max(min(value, max_value), min_value)


def grid_gen(width: int, height: int):
    return (expr.GridX(width, height), expr.GridY(width, height))


def regular_grid(
    min_x: expr.Num, max_x: expr.Num, min_y: expr.Num, max_y: expr.Num, n: int
):
    diff_x = max_x - min_x
    diff_y = max_y - min_y

    grid_x, grid_y = grid_gen(n, n)

    return (grid_x / n + 0.5) * diff_x + min_x, (grid_y / n + 0.5) * diff_y + min_y


def random_uniform(low: expr.Num, high: expr.Num, dim: int):
    return expr.Random(dim, expr.as_expr(low), expr.as_expr(high))


def debug(tag, orig):
    return expr.Debug(tag, orig)
