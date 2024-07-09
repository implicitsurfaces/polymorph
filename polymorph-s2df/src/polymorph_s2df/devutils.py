import jax.numpy as jnp
import matplotlib.pyplot as plt
from polymorph_num.expr import Expr, Num
from polymorph_num.unit import Unit
from polymorph_num.vec import Vec2

from .operations import Shape
from .paths import PathSegment


def p(x, y):
    return Vec2(x, y)


def eval_expr(expr: Expr):
    return Unit().register("result", expr).compile().evaluate("result")


def eval_distance(shape, x, y):
    return eval_expr(shape.distance(x, y))


def render_distance(shape: Shape, bounds=(-3, 3), n=500):
    x = jnp.linspace(bounds[0], bounds[1], n)
    X, Y = jnp.meshgrid(x, x)

    values = eval_distance(shape, X.ravel(), Y.ravel()).reshape(n, n)

    _, ax2 = plt.subplots(layout="constrained")

    levels = jnp.linspace(-5, 5, 41)

    ax2.axis("equal")
    ax2.contourf(
        X,
        Y,
        values,
        levels=levels,
        cmap="PRGn",
        origin="lower",
        extent=[bounds[0], bounds[1], bounds[0], bounds[1]],
    )
    ax2.contour(
        X,
        Y,
        values,
        levels=levels,
        colors="k",
        origin="lower",
        extent=[bounds[0], bounds[1], bounds[0], bounds[1]],
    )


def render(shape: Shape, bounds=(-3, 3), n=500):
    x = jnp.linspace(bounds[0], bounds[1], n)
    X, Y = jnp.meshgrid(x, x)

    values = eval_distance(shape, X.ravel(), Y.ravel()).reshape(n, n)

    plt.imshow(
        values > 0,
        cmap="gray",
        origin="lower",
        extent=(bounds[0], bounds[1], bounds[0], bounds[1]),
    )


class S(Shape):
    """A shape adapter for segments"""

    def __init__(self, segment: PathSegment):
        self.segment = segment

    def astuple(self):
        return (self.segment,)

    def distance(self, x: Num, y: Num) -> Expr:
        distance, _ = self.segment.distance_and_mask(Vec2(x, y))
        return distance


def compute_winding_number(segment, X, Y):
    return eval_expr(segment.winding_number(Vec2(X.ravel(), Y.ravel())))


def render_winding_number(segment: PathSegment, bounds=(-3, 3), n=500):
    x = jnp.linspace(bounds[0], bounds[1], n)
    X, Y = jnp.meshgrid(x, x)
    values = compute_winding_number(segment, X, Y).reshape(n, n)
    plt.imshow(
        values,
        origin="lower",
        cmap="coolwarm",
        extent=(bounds[0], bounds[1], bounds[0], bounds[1]),
    )
    plt.colorbar()
