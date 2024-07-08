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
    return expr.broadcast_binary(a, b, expr.BinOp.Min)


def max(a, b):
    return expr.broadcast_binary(a, b, expr.BinOp.Max)


def atan2(a, b):
    return expr.broadcast_binary(a, b, expr.BinOp.ArcTan2)


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


def debug(tag, orig):
    return expr.Debug(tag, orig)
