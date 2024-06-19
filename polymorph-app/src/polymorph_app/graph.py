import time
from functools import reduce
from typing import Any, Callable, FrozenSet, List, Self, Tuple

import jax.numpy as jnp
import polymorph_s2df as s2df

Expr = float  # TODO: Replace with a real expression type from polymorph-num
Point = Tuple[Expr, Expr]


class Node:
    def classname(self):
        return self.__class__.__name__


class Graph(Node):
    nodes: FrozenSet[Node] = frozenset()

    def add(self, node_ctor: Callable[[Self], Node]):
        shape = node_ctor()
        self.nodes |= frozenset([shape])
        return shape

    def to_sdf(self):
        return reduce(
            lambda a, b: s2df.Union(a, b),
            map(lambda s: s.to_sdf(), self.nodes),
            s2df.Circle(0),
        )


class Shape(Node):
    pass


class Circle(Node):
    center: Point = (0.0, 0.0)
    radius: Expr = 0.0

    def to_sdf(self):
        return s2df.Circle(self.radius).translate(jnp.array(self.center))

    def adjust(self, p1: Point, p2: Point):
        v1, v2 = jnp.array(p1), jnp.array(p2)
        self.center = p1
        self.radius = jnp.linalg.norm(v2 - v1)


class Rect(Node):
    p1: Point = (0.0, 0.0)
    p2: Point = (0.0, 0.0)

    def to_sdf(self):
        v1, v2 = jnp.array(self.p1), jnp.array(self.p2)
        center = (v1 + v2) / 2
        w, h = jnp.abs(v2 - v1)
        return s2df.Box(w, h).translate(center)

    def adjust(self, p1: Point, p2: Point):
        self.p1 = p1
        self.p2 = p2


class Polygon(Node):
    points: List[Point] = []

    def to_sdf(self):
        return (
            s2df.polygon(jnp.array(self.points))
            if len(self.points) > 2
            else s2df.Circle(0)
        )
