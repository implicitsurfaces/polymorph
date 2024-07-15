from dataclasses import dataclass, field
from functools import cached_property

import polymorph_s2df as s2df
from polymorph_num.expr import ZERO, Expr, Param, as_expr
from polymorph_num.ops import observation, param
from polymorph_num.vec import Vec2


@dataclass(frozen=True)
class BoundAtom:
    name: str

    def as_expr(self) -> Expr:
        return observation(self.name)


@dataclass(frozen=True)
class LockedAtom:
    val: float

    def as_expr(self) -> Expr:
        return as_expr(self.val)


@dataclass(frozen=True)
class FreeAtom:
    param: Param = field(default_factory=param)

    def as_expr(self) -> Expr:
        return self.param


type Atom = BoundAtom | LockedAtom | FreeAtom


class Shape:
    def classname(self):
        return self.__class__.__name__

    def to_sdf(self):
        raise NotImplementedError()

    def loss(self) -> Expr:
        return ZERO


class Value:
    """
    Shapes often have values associated with them. These values have identity
    and are owned by the shapes — i.e., a Circle's `center` value is unique
    and belongs to it.

    If a value is locked, its numeric attributes should be treated as fixed.
    If its unlocked, its numeric attributes are free to vary and are just
    caching their most recent values.
    """

    locked: bool

    def __init__(self):
        self.locked = False


class PointValue:
    x: Atom
    y: Atom

    def __init__(self):
        self.x = FreeAtom()
        self.y = FreeAtom()

    @cached_property
    def param_expr(self) -> Vec2:
        return Vec2(param(), param())

    def lock(self, x: float, y: float) -> None:
        self.x = LockedAtom(x)
        self.y = LockedAtom(y)

    def bind(self, x: str, y: str) -> None:
        self.x = BoundAtom(x)
        self.y = BoundAtom(y)

    def free(self):
        self.x = FreeAtom()
        self.y = FreeAtom()

    def as_vec2(self) -> Vec2:
        return Vec2(self.x.as_expr(), self.y.as_expr())


class LengthValue(Value):
    mm: Atom

    def __init__(self):
        self.mm = FreeAtom()

    def lock(self, mm: float) -> None:
        assert isinstance(self.mm, FreeAtom)
        self.mm = LockedAtom(mm)

    def bind(self, mm: str) -> None:
        assert isinstance(self.mm, FreeAtom)
        self.mm = BoundAtom(mm)

    def free(self) -> None:
        self.mm = FreeAtom()

    def as_expr(self) -> Expr:
        return self.mm.as_expr()


class AreaValue(Value):
    mm2: float


class AngleValue(Value):
    degrees: float


class Constraint:
    """
    Constraints relate values to shapes and to each other. The values
    appearing here could be those belonging to shapes, or could be
    owned by the constraint (usually/always with locked values).
    """

    def loss(self) -> Expr:
        raise NotImplementedError()


@dataclass
class EqualLengthConstraint(Constraint):
    a: LengthValue
    b: LengthValue


@dataclass
class DistanceConstraint(Constraint):
    a: PointValue
    b: PointValue
    length: LengthValue


@dataclass
class AreaConstraint(Constraint):
    shape: Shape
    area: AreaValue


@dataclass
class CentroidConstraint(Constraint):
    shape: Shape
    centroid: PointValue


@dataclass
class OnBoundaryConstraint(Constraint):
    shape: Shape
    point: PointValue

    def loss(self) -> Expr:
        p = self.point.as_vec2()
        d = self.shape.to_sdf().distance(p.x, p.y)
        return d * d


class AngleConstraint(Constraint):
    a: PointValue
    b: PointValue
    c: PointValue
    angleABC: AngleValue


class Sketch:
    shapes: list[Shape]
    constraints: list[Constraint]

    def __init__(self):
        self.shapes = []
        self.constraints = []
        self._sdfs = None

    def __iter__(self):
        return iter(self.shapes)

    @property
    def cached_sdfs(self) -> tuple[s2df.Shape]:
        if self._sdfs is None:
            self._sdfs = tuple(n.to_sdf() for n in self.shapes)
        return self._sdfs

    def changed(self):
        self._sdfs = None

    def add(self, node):
        self.shapes.append(node)
        self.changed()
        return node

    def total_loss(self) -> Expr:
        return as_expr(sum(c.loss() for c in self.constraints))


class Circle(Shape):
    center: PointValue
    radius: LengthValue

    __match_args__ = ("center", "radius")

    def __init__(self):
        self.center = PointValue()
        self.radius = LengthValue()
        self._boundary_points = []

    def boundary_point(self) -> PointValue:
        p = PointValue()
        self._boundary_points.append(p)
        return p

    def to_sdf(self):
        return s2df.Circle(self.radius.as_expr()).translate(self.center.as_vec2())


class Box(Shape):
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


class Polygon(Shape):
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
