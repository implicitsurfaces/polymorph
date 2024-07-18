import math
from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable, Self

import polymorph_s2df as s2df
from polymorph_num.expr import PI, ZERO, Expr, Param, as_expr
from polymorph_num.ops import atan2, observation, param
from polymorph_num.unit import ParamValues
from polymorph_num.vec import Vec2
from polymorph_s2df import geometric_properties


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


@dataclass
class PointValue:
    x: Atom = field(default_factory=FreeAtom)
    y: Atom = field(default_factory=FreeAtom)

    @cached_property
    def param_expr(self) -> Vec2:
        return Vec2(param(), param())

    def lock(self, x: float = 0.0, y: float = 0.0) -> Self:
        self.x = LockedAtom(x)
        self.y = LockedAtom(y)
        return self

    def bind(self, x: str, y: str) -> Self:
        self.x = BoundAtom(x)
        self.y = BoundAtom(y)
        return self

    def free(self) -> Self:
        self.x = FreeAtom()
        self.y = FreeAtom()
        return self

    def as_vec2(self) -> Vec2:
        return Vec2(self.x.as_expr(), self.y.as_expr())


class LengthValue(Value):
    mm: Atom

    def __init__(self):
        self.mm = FreeAtom()

    def lock(self, mm: float = 0.0) -> Self:
        self.mm = LockedAtom(mm)
        return self

    def lock_from_computed(self, param_values: ParamValues):
        if not isinstance(self.mm, FreeAtom):
            raise ValueError("Can only lock a free value from a computed one")

        raw_param = param_values.get(self.mm.param)
        self.lock(math.exp(raw_param))
        return self

    def bind(self, mm: str) -> Self:
        self.mm = BoundAtom(mm)
        return self

    def free(self) -> Self:
        self.mm = FreeAtom()
        return self

    def as_expr(self) -> Expr:
        if isinstance(self.mm, FreeAtom):
            return self.mm.as_expr().exp()
        return self.mm.as_expr()


class AreaValue(Value):
    mm2: Atom

    def __init__(self):
        self.mm2 = FreeAtom()

    def lock(self, mm2: float = 0.0) -> None:
        self.mm2 = LockedAtom(mm2)

    def lock_from_computed(self, param_values: ParamValues):
        if not isinstance(self.mm2, FreeAtom):
            raise ValueError("Can only lock a free value from a computed one")

        raw_param = param_values.get(self.mm2.param)
        self.lock(math.exp(raw_param))
        return self

    def bind(self, mm2: str) -> None:
        self.mm2 = BoundAtom(mm2)

    def free(self) -> None:
        self.mm2 = FreeAtom()

    def as_expr(self) -> Expr:
        if isinstance(self.mm2, FreeAtom):
            return self.mm2.as_expr().exp()
        return self.mm2.as_expr()


class AngleValue(Value):
    degrees: Atom

    def __init__(self):
        self.degrees = FreeAtom()

    def lock(self, degrees: float = 0.0) -> None:
        self.degrees = LockedAtom(degrees)

    def lock_from_computed(self, param_values: ParamValues):
        if not isinstance(self.degrees, FreeAtom):
            raise ValueError("Can only lock a free value from a computed one")

        raw_param = param_values.get(self.degrees.param)
        self.lock(math.atan(raw_param) * 180 / math.pi)
        return self

    def bind(self, degrees: str) -> None:
        self.degrees = BoundAtom(degrees)

    def free(self) -> None:
        self.degrees = FreeAtom()

    def as_expr(self) -> Expr:
        if isinstance(self.degrees, FreeAtom):
            # When exposing a free angle, we expose it to the optimizer as the tangent
            # (so that it is defined on the whole real line)
            return self.degrees.as_expr().atan()

        # We store the angle in degrees, but convert to radians for the compute graph
        return self.degrees.as_expr() / 180 * PI


class MathValue(Value):
    orig: Value
    fun: Callable[[Expr], Expr]

    def __init__(self, orig: Value, fun: Callable[[Expr], Expr]):
        self.orig = orig
        self.fun = fun

    def as_expr(self) -> Expr:
        return self.fun(self.orig.as_expr())


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
        shape_area = geometric_properties.area(shape, size)

        diff = self.area.as_expr() - shape_area
        return diff * diff


@dataclass
class CentroidConstraint(Constraint):
    shape: Shape
    centroid: PointValue

    def loss(self, **kwargs) -> Expr:
        size = kwargs.get("size", (100, 100))
        shape = self.shape.to_sdf()
        centroid = geometric_properties.centroid(shape, size)

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


@dataclass
class VerticallyAligned(Constraint):
    p1: PointValue
    p2: PointValue

    def loss(self, **kwargs) -> Expr:
        d = self.p1.as_vec2().y - self.p2.as_vec2().y
        return d * d


@dataclass
class SameValueConstraint(Constraint):
    a: Value
    b: Value

    def loss(self, **kwargs) -> Expr:
        diff = self.a.as_expr() - self.b.as_expr()
        return diff * diff


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

    def add(self, shape: Shape) -> None:
        self.shapes.append(shape)
        self.changed()

    def add_constraint(self, c: Constraint) -> None:
        self.constraints.append(c)
        self.changed()

    def remove_constraint(self, c: Constraint) -> None:
        self.constraints.remove(c)
        self.changed()

    def total_loss(self, **kwargs) -> Expr:
        return as_expr(sum(c.loss(**kwargs) for c in self.constraints))


class Centroid(PointValue):
    shape: Shape
    size: tuple[int, int]

    def __init__(self, shape: Shape, size: tuple[int, int]):
        self.shape = shape
        self.size = size

    def as_vec2(self) -> Vec2:
        return geometric_properties.centroid_monte_carlo(
            self.shape.to_sdf(), self.size, 10000
        )


class Area(Value):
    shape: Shape
    size: tuple[int, int]

    def __init__(self, shape: Shape, size: tuple[int, int]):
        self.shape = shape
        self.size = size

    def as_expr(self) -> Expr:
        return geometric_properties.area_monte_carlo(
            self.shape.to_sdf(), self.size, 10000
        )


class Circle(Shape):
    center: PointValue
    radius: LengthValue

    @property
    def position(self) -> PointValue:
        return self.center

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
    p1: PointValue
    p2: PointValue

    position: PointValue
    rotation: AngleValue

    def __init__(self):
        self.p1 = PointValue()
        self.p2 = PointValue()

        self.rotation = AngleValue()
        self.position = PointValue()

    def to_sdf(self):
        v1, v2 = self.p1.as_vec2(), self.p2.as_vec2()
        center = (v1 + v2) / 2
        w = (v2.x - v1.x).smoothabs()
        h = (v2.y - v1.y).smoothabs()
        return s2df.Box(w, h).translate(center)


class CenteredBox(Shape):
    width: LengthValue
    height: LengthValue

    position: PointValue
    rotation: AngleValue

    def __init__(self):
        self.width = LengthValue()
        self.height = LengthValue()

        self.rotation = AngleValue()
        self.position = PointValue()

    def to_sdf(self):
        return (
            s2df.Box(self.width.as_expr(), self.height.as_expr())
            .rotate(self.rotation.as_expr())
            .translate(self.position.as_vec2())
        )


class Polygon(Shape):
    points: list[PointValue]
    temp_point: PointValue | None = None

    position: PointValue
    rotation: AngleValue

    def __init__(self):
        super().__init__()
        self.points = []
        self.position = PointValue()
        self.rotation = AngleValue()

    def add_point(self) -> PointValue:
        p = PointValue()
        self.points.append(p)
        return p

    def to_sdf(self):
        points = self.points + ([self.temp_point] if self.temp_point else [])
        if len(points) < 3:
            return s2df.Circle(0.0)
        segments = [
            s2df.LineSegment(
                points[i].as_vec2(),
                points[(i + 1) % len(points)].as_vec2(),
            )
            for i in range(len(points))
        ]
        return (
            s2df.ClosedPath(segments)
            .rotate(self.rotation.as_expr())
            .translate(self.position.as_vec2())
        )


class TopHalfPlane(Shape):
    position: PointValue
    rotation: AngleValue

    def __init__(self):
        self.position = PointValue()
        self.rotation = AngleValue()

    def to_sdf(self):
        return (
            s2df.TopHalfPlane()
            .rotate(self.rotation.as_expr())
            .translate(self.position.as_vec2())
        )


class BottomHalfPlane(Shape):
    position: PointValue
    rotation: AngleValue

    def __init__(self):
        self.position = PointValue()
        self.rotation = AngleValue()

    def to_sdf(self):
        return (
            s2df.BottomHalfPlane()
            .rotate(self.rotation.as_expr())
            .translate(self.position.as_vec2())
        )


class LeftHalfPlane(Shape):
    position: PointValue
    rotation: AngleValue

    def __init__(self):
        self.position = PointValue()
        self.rotation = AngleValue()

    def to_sdf(self):
        return (
            s2df.LeftHalfPlane()
            .rotate(self.rotation.as_expr())
            .translate(self.position.as_vec2())
        )


class RightHalfPlane(Shape):
    position: PointValue
    rotation: AngleValue

    def __init__(self):
        self.position = PointValue()
        self.rotation = AngleValue()

    def to_sdf(self):
        return (
            s2df.RightHalfPlane()
            .rotate(self.rotation.as_expr())
            .translate(self.position.as_vec2())
        )


class Union(Shape):
    a: Shape
    b: Shape

    def __init__(self, a: Shape, b: Shape):
        self.a = a
        self.b = b

    def to_sdf(self):
        return self.a.to_sdf().union(self.b.to_sdf())


class Intersection(Shape):
    a: Shape
    b: Shape

    def __init__(self, a: Shape, b: Shape):
        self.a = a
        self.b = b

    def to_sdf(self):
        return self.a.to_sdf().intersect(self.b.to_sdf())


class Difference(Shape):
    a: Shape
    b: Shape

    def __init__(self, a: Shape, b: Shape):
        self.a = a
        self.b = b

    def to_sdf(self):
        return self.a.to_sdf().substract(self.b.to_sdf())


class Shell(Shape):
    shape: Shape
    thickness: LengthValue

    def __init__(self, shape: Shape):
        self.shape = shape
        self.thickness = LengthValue()

    def to_sdf(self):
        return self.shape.to_sdf().shell(self.thickness.as_expr())


class Scale(Shape):
    shape: Shape
    factor: LengthValue

    def __init__(self, shape: Shape):
        self.shape = shape
        self.factor = LengthValue()

    def to_sdf(self):
        return self.shape.to_sdf().scale(self.factor.as_expr())


class Morph(Shape):
    a: Shape
    b: Shape
    t: LengthValue

    def __init__(self, a: Shape, b: Shape):
        self.a = a
        self.b = b
        self.t = LengthValue()

    def to_sdf(self):
        return self.a.to_sdf().morph(self.t.as_expr(), self.b.to_sdf())
