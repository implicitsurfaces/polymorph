from dataclasses import dataclass, field
from functools import cached_property

import polymorph_s2df as s2df
from polymorph_num.expr import PI, ZERO, Expr, Param, as_expr
from polymorph_num.ops import atan2, observation, param
from polymorph_num.vec import Vec2
from polymorph_s2df import routines


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

    def as_expr(self) -> Expr:
        raise NotImplementedError()


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
    mm2: Atom

    def __init__(self):
        self.mm2 = FreeAtom()

    def lock(self, mm2: float) -> None:
        assert isinstance(self.mm2, FreeAtom)
        self.mm2 = LockedAtom(mm2)

    def bind(self, mm2: str) -> None:
        assert isinstance(self.mm2, FreeAtom)
        self.mm2 = BoundAtom(mm2)

    def free(self) -> None:
        self.mm2 = FreeAtom()

    def as_expr(self) -> Expr:
        return self.mm2.as_expr()


class AngleValue(Value):
    degrees: Atom

    def __init__(self):
        self.mm2 = FreeAtom()

    def lock(self, mm2: float) -> None:
        assert isinstance(self.mm2, FreeAtom)
        self.mm2 = LockedAtom(mm2)

    def bind(self, mm2: str) -> None:
        assert isinstance(self.mm2, FreeAtom)
        self.mm2 = BoundAtom(mm2)

    def free(self) -> None:
        self.mm2 = FreeAtom()

    def as_expr(self) -> Expr:
        if isinstance(self.mm2, FreeAtom):
            # When exposing a free angle, we expose it to the optimizer as the tangent
            # (so that it is defined on the whole real line)
            return self.mm2.as_expr().atan()

        # We store the angle in degrees, but convert to radians for the compute graph
        return self.mm2.as_expr() / 180 * PI


class Constraint:
    """
    Constraints relate values to shapes and to each other. The values
    appearing here could be those belonging to shapes, or could be
    owned by the constraint (usually/always with locked values).
    """

    def loss(self, **kargs) -> Expr:
        raise NotImplementedError()


@dataclass
class EqualLengthConstraint(Constraint):
    a: LengthValue
    b: LengthValue

    def loss(self, **kwargs) -> Expr:
        diff = self.a.as_expr() - self.b.as_expr()
        return diff * diff


@dataclass
class DistanceConstraint(Constraint):
    a: PointValue
    b: PointValue
    length: LengthValue

    def loss(self, **kwargs) -> Expr:
        diff = (self.a.as_vec2() - self.b.as_vec2()).norm() - self.length.as_expr()
        return diff * diff


@dataclass
class AreaConstraint(Constraint):
    shape: Shape
    area: AreaValue

    def loss(self, **kwargs) -> Expr:
        size = kwargs.get("size", (100, 100))
        shape = self.shape.to_sdf()
        shape_area = routines.area(shape, size)

        diff = self.area.as_expr() - shape_area
        return diff * diff


@dataclass
class CentroidConstraint(Constraint):
    shape: Shape
    centroid: PointValue

    def loss(self, **kwargs) -> Expr:
        size = kwargs.get("size", (100, 100))
        shape = self.shape.to_sdf()
        centroid = routines.centroid(shape, size)

        diff = self.centroid.as_vec2() - centroid
        return diff.dot(diff)  # The square of the norm


@dataclass
class OnBoundaryConstraint(Constraint):
    shape: Shape
    point: PointValue

    def loss(self, **kwargs) -> Expr:
        p = self.point.as_vec2()
        d = self.shape.to_sdf().distance(p.x, p.y)
        return d * d


class AngleConstraint(Constraint):
    a: PointValue
    b: PointValue
    c: PointValue
    angleABC: AngleValue

    def loss(self, **kwargs) -> Expr:
        a = self.a.as_vec2()
        b = self.b.as_vec2()
        c = self.c.as_vec2()

        ab = b - a
        bc = c - b

        angle = atan2(ab.dot(bc), ab.cross(bc))

        diff = angle - self.angleABC.as_expr()
        return diff * diff


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

    def total_loss(self, **kwargs) -> Expr:
        return as_expr(sum(c.loss(**kwargs) for c in self.constraints))


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
