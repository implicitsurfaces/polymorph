from dataclasses import dataclass

from polymorph_num.expr import Expr, as_expr
from polymorph_num.ops import minimum


def norm(x, y) -> Expr:
    return (x * x + y * y).sqrt()


@dataclass(frozen=True)
class Shape:
    def is_inside(self, x: Expr, y: Expr, scale=100) -> Expr:
        return as_expr(1) - (self.distance(x, y) * scale).sigmoid()

    def distance(self, x: Expr, y: Expr) -> Expr:
        raise NotImplementedError()


@dataclass(frozen=True)
class Translation(Shape):
    x: Expr
    y: Expr
    shape: Shape

    def distance(self, x: Expr, y: Expr) -> Expr:
        return self.shape.distance(x - self.x, y - self.y)


@dataclass(frozen=True)
class Circle(Shape):
    radius: Expr

    def distance(self, x: Expr, y: Expr) -> Expr:
        return norm(x, y) - self.radius


@dataclass(frozen=True)
class Union(Shape):
    a: Shape
    b: Shape

    def distance(self, x: Expr, y: Expr) -> Expr:
        return minimum(self.a.distance(x, y), self.b.distance(x, y))


# @dataclass(frozen=True)
# class Box(Shape):
#     def __init__(self, width, height):
#         self.width = width
#         self.height = height

#     @property
#     def center(self):
#         return jnp.array([self.width, self.height]) / 2

#     def distance(self, p):
#         q = jnp.abs(p) - self.center
#         return length(soft_plus(q)) + soft_minus(jnp.amax(q, axis=1))
