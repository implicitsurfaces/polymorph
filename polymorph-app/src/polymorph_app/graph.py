from functools import cached_property
from typing import FrozenSet

import polymorph_s2df as sdf
from polymorph_num.expr import Expr, Observation, Param, as_expr
from polymorph_num.vec import Vec2


class Node:
    def classname(self):
        return self.__class__.__name__

    def to_sdf(self):
        raise NotImplementedError

    def loss(self) -> Expr:
        return as_expr(0.0)


class Graph(Node):
    nodes: FrozenSet[Node] = frozenset()
    last_node: Node | None = None

    @cached_property
    def cached_sdf(self):
        return self.to_sdf()

    def changed(self):
        if hasattr(self, "cached_sdf"):
            del self.cached_sdf

    def add(self, node):
        self.nodes |= frozenset([node])
        self.last_node = node
        self.changed()
        return node

    def to_sdf(self):
        ans = sdf.Circle(as_expr(0.0))
        for s in self.nodes:
            ans = ans.union(s.to_sdf())
        return ans

    def total_loss(self) -> Expr:
        ans = as_expr(0.0)
        for n in self.nodes:
            ans += n.loss()
        return ans


class Shape(Node):
    pass


class Circle(Node):
    center: Vec2
    radius: Expr

    __match_args__ = ("center", "radius")

    def __init__(self, center: Vec2, radius: Expr):
        self.center = center
        self.radius = radius

    def to_sdf(self):
        return sdf.Circle(self.radius).translate(self.center)

    def loss(self) -> Expr:
        if isinstance(self.radius, Param):
            d = self.to_sdf().distance(Observation("mouse_x"), Observation("mouse_y"))
            return d * d
        return as_expr(0.0)


def as_vec2(p):
    return Vec2(*p.tolist())


class Box(Node):
    p1: Vec2
    p2: Vec2

    __match_args__ = ("p1", "p2")

    def __init__(self, p1: Vec2, p2: Vec2):
        self.p1 = p1
        self.p2 = p2

    def to_sdf(self):
        center = (self.p1 + self.p2) / 2
        w = (self.p2.x - self.p1.x).smoothabs()
        h = (self.p2.y - self.p1.y).smoothabs()
        return sdf.Box(w, h).translate(center)


class Polygon(Node):
    points: list[Vec2] = []

    __match_args__ = "points"

    def to_sdf(self):
        if len(self.points) < 3:
            return sdf.Circle(as_expr(0.0))
        segments = [
            sdf.LineSegment(
                as_vec2(self.points[i]),
                as_vec2(self.points[(i + 1) % len(self.points)]),
            )
            for i in range(len(self.points))
        ]
        return sdf.ClosedPath(tuple(segments))
