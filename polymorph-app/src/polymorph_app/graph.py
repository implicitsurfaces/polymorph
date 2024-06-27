from typing import Callable, FrozenSet, List

from polymorph_num import loss
from polymorph_num.expr import Expr, as_expr
from polymorph_num.vec import Vec2

from . import sdf


class Node:
    def classname(self):
        return self.__class__.__name__

    def to_sdf(self):
        raise NotImplementedError


class Graph(Node):
    nodes: FrozenSet[Node] = frozenset()
    _sdf: sdf.Shape | None = None

    def changed(self):
        self._sdf = None

    def add(self, node_ctor: Callable[[], Node]):
        shape = node_ctor()
        self.nodes |= frozenset([shape])
        self.changed()
        return shape

    def to_sdf(self):
        if self._sdf:
            return self._sdf

        ans = sdf.Circle(as_expr(0.0))
        for s in self.nodes:
            ans = sdf.Union(ans, s.to_sdf())
        self._sdf = ans
        return ans

    def total_loss(self):
        return loss.Loss(as_expr(0))  # TODO


class Shape(Node):
    pass


class Circle(Node):
    center = Vec2(0.0, 0.0)
    radius: Expr = as_expr(0.0)

    __match_args__ = ("center", "radius")

    def to_sdf(self):
        c = self.center
        return sdf.Translation(c.x, c.y, sdf.Circle(self.radius))

    def adjust(self, p1, p2):
        self.center = Vec2(*p1.tolist())
        self.radius = (Vec2(*p2.tolist()) - Vec2(*p1.tolist())).norm()


class Box(Node):
    p1 = Vec2(0.0, 0.0)
    p2 = Vec2(0.0, 0.0)

    __match_args__ = ("p1", "p2")

    def to_sdf(self):
        raise NotImplementedError()

    def adjust(self, p1, p2):
        self.p1 = Vec2(*p1.tolist())
        self.p2 = Vec2(*p2.tolist())


class Polygon(Node):
    points: List[Vec2] = []

    __match_args__ = "points"

    def to_sdf(self):
        raise NotImplementedError()
