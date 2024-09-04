import textwrap
from typing import Iterable

from polymorph_num import ops
from polymorph_num.expr import PI, TAU, ZERO, Expr, Infinity
from polymorph_num.vec import Vec2


def indent_shape(shape):
    return textwrap.indent(repr(shape), "  ")


def repr_point(p):
    return f"p({p.x}, {p.y})"


def norm(x: Expr, y: Expr) -> Expr:
    return (x * x + y * y).sqrt()


def smooth_clamp_mask(x: Expr, low=0.0, high=1.0, softness=1e-6):
    """A smooth implementation of a mask between the boundary values"""
    lower_transition = 0.5 * (1 + ((x - low) / softness).tanh())
    upper_transition = 0.5 * (1 - ((x - high) / softness).tanh())
    return lower_transition * upper_transition


def sum_iterable(values: Iterable[Expr]) -> Expr:
    x = ZERO
    for item in values:
        x += item
    return x


def min_iterable(values: Iterable[Expr]) -> Expr:
    x = Infinity
    for item in values:
        x = ops.min(x, item)
    return x


def max_iterable(values: Iterable[Expr]) -> Expr:
    x = -Infinity
    for item in values:
        x = ops.max(x, item)
    return x


def normalize_angle(q):
    """
    Normalize an angle to the range [0, 2π)
    """
    return ((q % TAU) + TAU) % TAU


def normalize_angle_signed(q):
    """
    Normalize an angle to the range [-π, π)
    """
    return ((q + PI) % TAU) - PI


def angular_distance(start_angle, end_angle, orientation_sign):
    raw_distance = orientation_sign * (end_angle - start_angle)
    return (raw_distance + TAU) % TAU


def min_non_zero(a: Expr, b: Expr):
    min_value = ops.min(a, b)
    return ops.if_eq(min_value, ZERO, ops.max(a, b), min_value)


def max_non_zero(a: Expr, b: Expr):
    max_value = ops.max(a, b)
    return ops.if_eq(max_value, ZERO, ops.min(a, b), max_value)


def diamond_atan(x, y):
    sign_x = x.sign()
    sign_y = y.sign()

    x = x.abs()
    y = y.abs()

    denom = x + y

    q1 = y / denom
    q2 = 1 + x / denom
    q3 = 2 + y / denom
    q4 = 3 + x / denom

    is_q1 = (1 + sign_y) * (1 + sign_x)
    is_q2 = (1 + sign_y) * (1 - sign_x)
    is_q3 = (1 - sign_y) * (1 - sign_x)
    is_q4 = (1 - sign_y) * (1 + sign_x)

    return 0.25 * (q1 * is_q1 + q2 * is_q2 + q3 * is_q3 + q4 * is_q4)


def diamond_tan(diangle):
    x_q1_q2 = 1 - diangle
    x_q2_q3 = diangle - 3

    y_q1 = diangle
    y_q2 = 2 - diangle
    y_q3_q4 = diangle - 4

    x_val = ops.if_lt(diangle, 2, x_q1_q2, x_q2_q3)
    y_val = ops.if_lt(diangle, 3, ops.if_gt(diangle, 1, y_q2, y_q1), y_q3_q4)

    return Vec2(x_val, y_val).normalized()
