import textwrap
from typing import Iterable

from polymorph_num import ops
from polymorph_num.expr import ZERO, Expr, Infinity


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
