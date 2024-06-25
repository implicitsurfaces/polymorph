import time
from functools import reduce
from typing import Any, Callable, FrozenSet, List, Self, Tuple

import jax.numpy as jnp
import polymorph_s2df as s2df
from polymorph_num import loss, ops
from polymorph_num.expr import Expr, as_expr
from polymorph_num.point import Point


class Node:
    def classname(self):
        return self.__class__.__name__

    def register_outputs(self, loss):
        pass


class Graph(Node):
    nodes: FrozenSet[Node] = frozenset()

    def add(self, node_ctor: Callable[[Self], Node]):
        shape = node_ctor()
        self.nodes |= frozenset([shape])
        return shape

    def to_sdf(self, soln):
        ans = s2df.Circle(0)
        for s in self.nodes:
            ans = s2df.Union(ans, s.to_sdf(soln))
        return ans

    def register_outputs(self, loss):
        for n in self.nodes:
            n.register_outputs(loss)

    def total_loss(self):
        l = loss.Loss(as_expr(0))  # TODO
        for n in self.nodes:
            n.register_outputs(l)
        return l


class Shape(Node):
    pass


class Circle(Node):
    center = Point(0.0, 0.0)
    radius: Expr = as_expr(0.0)

    __match_args__ = ("center", "radius")

    def to_sdf(self, soln):
        r = soln.eval(self.radius)
        c = soln.eval(self.center)
        return s2df.Circle(r).translate(c)

    def adjust(self, p1, p2):
        self.center = Point(*p1.tolist())
        self.radius = (Point(*p2.tolist()) - Point(*p1.tolist())).length()

    def register_outputs(self, loss):
        loss.register_output(self.center)
        loss.register_output(self.radius)


class Box(Node):
    p1 = Point(0.0, 0.0)
    p2 = Point(0.0, 0.0)

    __match_args__ = ("p1", "p2")

    def to_sdf(self, soln):
        w, h, center = (soln.eval(o) for o in self._outputs)
        return s2df.Box(w, h).translate(center)

    def adjust(self, p1, p2):
        self.p1 = Point(*p1.tolist())
        self.p2 = Point(*p2.tolist())

    def register_outputs(self, loss):
        center = Point.origin() + (self.p1 + self.p2) / 2
        w = ops.smoothabs(self.p2.x - self.p1.x)
        h = ops.smoothabs(self.p2.y - self.p1.y)
        self._outputs = [w, h, center]
        for o in self._outputs:
            loss.register_output(o)


class Polygon(Node):
    points: List[Point] = []

    __match_args__ = "points"

    def to_sdf(self, soln):
        return (
            s2df.polygon(jnp.array(self.points))
            if len(self.points) > 2
            else s2df.Circle(0)
        )
