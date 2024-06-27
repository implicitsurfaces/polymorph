from dataclasses import dataclass

from polymorph_num import ops
from polymorph_num.expr import Expr, as_expr
from polymorph_num.vec import Vec2


def norm(x: Expr, y: Expr) -> Expr:
    return (x * x + y * y).sqrt()


def softminus(x: Expr) -> Expr:
    return x - x.softplus()


@dataclass(frozen=True)
class Shape:
    def is_inside(self, x: Expr, y: Expr, scale=100) -> Expr:
        return as_expr(1) - (self.distance(x, y) * scale).sigmoid()

    def distance(self, x: Expr, y: Expr) -> Expr:
        raise NotImplementedError()


@dataclass(frozen=True)
class Translation(Shape):
    offset: Vec2
    shape: Shape

    def distance(self, x: Expr, y: Expr) -> Expr:
        return self.shape.distance(x - self.offset.x, y - self.offset.y)


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
        return ops.min(self.a.distance(x, y), self.b.distance(x, y))


@dataclass(frozen=True)
class Box(Shape):
    width: Expr
    height: Expr

    def distance(self, x: Expr, y: Expr) -> Expr:
        q_x = x.smoothabs() - self.width / 2
        q_y = y.smoothabs() - self.height / 2

        return norm(q_x.softplus(), q_y.softplus()) + softminus(ops.max(q_x, q_y))
