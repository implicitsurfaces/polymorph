from typing import FrozenSet

import polymorph_s2df as s2df
from polymorph_num.expr import ZERO, Expr, Observation, Param, as_expr
from polymorph_num.vec import Vec2


class Node:
    def classname(self):
        return self.__class__.__name__

    def to_sdf(self):
        raise NotImplementedError

    def loss(self) -> Expr:
        return ZERO


class Graph(Node):
    nodes: FrozenSet[Node] = frozenset()
    last_node: Node | None = None

    def __init__(self):
        self._sdfs = None

    @property
    def cached_sdfs(self) -> tuple[s2df.Shape]:
        if self._sdfs is None:
            self._sdfs = tuple(n.to_sdf() for n in self.nodes)
        return self._sdfs

    def changed(self):
        self._sdfs = None

    def add(self, node):
        self.nodes |= frozenset([node])
        self.last_node = node
        self.changed()
        return node

    def total_loss(self) -> Expr:
        return as_expr(sum(n.loss() for n in self.nodes))


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
        return s2df.Circle(self.radius).translate(self.center)

    def loss(self) -> Expr:
        if isinstance(self.radius, Param):
            d = self.to_sdf().distance(Observation("mouse_x"), Observation("mouse_y"))
            return d * d
        return ZERO


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
        return s2df.Box(w, h).translate(center)


class Polygon(Node):
    points: list[Vec2]
    temp_point: Vec2 | None = None

    __match_args__ = "points"

    def __init__(self):
        self.points = []

    def to_sdf(self):
        points = self.points + ([self.temp_point] if self.temp_point else [])
        if len(points) < 3:
            return s2df.Circle(0.0)
        segments = [
            s2df.LineSegment(
                points[i],
                points[(i + 1) % len(points)],
            )
            for i in range(len(points))
        ]
        return s2df.ClosedPath(segments)
