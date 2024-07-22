from polymorph_num import ops
from polymorph_num.expr import Expr
from polymorph_num.vec import Vec2

from .utils import max_iterable, min_iterable


class BoundingBox:
    def __init__(self, min_x: Expr, min_y: Expr, max_x: Expr, max_y: Expr):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y

    @property
    def min(self):
        return Vec2(self.min_x, self.min_y)

    @property
    def max(self):
        return Vec2(self.max_x, self.max_y)

    def union(self, other):
        return BoundingBox(
            ops.min(self.min_x, other.min_x),
            ops.min(self.min_y, other.min_y),
            ops.max(self.max_x, other.max_x),
            ops.max(self.max_y, other.max_y),
        )

    def intersection(self, other):
        return BoundingBox(
            ops.max(self.min_x, other.min_x),
            ops.max(self.min_y, other.min_y),
            ops.min(self.max_x, other.max_x),
            ops.min(self.max_y, other.max_y),
        )

    def translate(self, x: Expr, y: Expr):
        return BoundingBox(
            self.min_x + x, self.min_y + y, self.max_x + x, self.max_y + y
        )

    def rotate(self, angle: Expr):
        p0 = Vec2(self.min_x, self.min_y)
        p1 = Vec2(self.max_x, self.min_y)
        p2 = Vec2(self.max_x, self.max_y)
        p3 = Vec2(self.min_x, self.max_y)

        p0 = p0.rotate(angle)
        p1 = p1.rotate(angle)
        p2 = p2.rotate(angle)
        p3 = p3.rotate(angle)

        return bounding_box_from_points([p0, p1, p2, p3])


def bounding_box_from_points(points: list[Vec2]):
    min_x = min_iterable(p.x for p in points)
    min_y = min_iterable(p.y for p in points)
    max_x = max_iterable(p.x for p in points)
    max_y = max_iterable(p.y for p in points)
    return BoundingBox(min_x, min_y, max_x, max_y)
