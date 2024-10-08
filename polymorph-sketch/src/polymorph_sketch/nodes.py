from dataclasses import dataclass
from typing import TypeAlias

PositiveFloat: TypeAlias = float
Sign: TypeAlias = int


@dataclass(frozen=True)
class Distance:
    def __add__(self, other: "Distance"):
        return DistanceSum(self, other)

    def __mul__(self, other: PositiveFloat):
        return DistanceScaled(self, other)

    def __rmul__(self, other: PositiveFloat):
        return DistanceScaled(self, other)


@dataclass(frozen=True)
class Point:
    def __add__(self, other: "Vector") -> "Point":
        if not isinstance(other, Vector):
            raise NotImplementedError
        return VectorPointSum(self, other)

    def __radd__(self, other: "Vector"):
        return self.__add__(other)

    def __sub__(self, other: "Point") -> "Vector":
        return VectorFromPoints(other, self)

    def vec(self) -> "Vector":
        """Return the vector from the origin to this point."""
        return VectorFromPoint(self)


@dataclass(frozen=True)
class Vector:
    def __add__(self, other: "Vector") -> "Vector":
        return VectorSum(self, other)

    def __radd__(self, other: "Vector"):
        return self.__add__(other)

    def __sub__(self, other: "Vector"):
        return VectorDifference(self, other)

    def direction(self) -> "Angle":
        return VectorDirection(self)

    def norm(self) -> "Distance":
        return VectorNorm(self)

    def from_origin(self) -> "Point":
        return VectorOriginSum(self)


@dataclass(frozen=True)
class Angle:
    def __add__(self, other: "Angle"):
        return AngleSum(self, other)

    def __sub__(self, other: "Angle"):
        return AngleDifference(self, other)

    def bisect(self):
        return AngleBisection(self)

    def perpendicular(self):
        return PerpendicularAngle(self)

    def opposite(self):
        return OppositeAngle(self)


@dataclass(frozen=True)
class Edge:
    pass


@dataclass(frozen=True)
class Path:
    pass


@dataclass(frozen=True)
class Shape:
    pass


@dataclass(frozen=True)
class Constraint:
    pass


@dataclass(frozen=True)
class DistanceLiteral(Distance):
    length: PositiveFloat


@dataclass(frozen=True)
class DistanceParam(Distance):
    def __eq__(self, other):
        return self is other


@dataclass(frozen=True)
class DistanceSum(Distance):
    left: Distance
    right: Distance


@dataclass(frozen=True)
class DistanceScaled(Distance):
    distance: Distance
    scale: PositiveFloat


@dataclass(frozen=True)
class VectorNorm(Distance):
    vector: Vector


@dataclass(frozen=True)
class ArcLength(Distance):
    angle: Angle
    radius: Distance


@dataclass(frozen=True)
class AngleLiteral(Angle):
    degrees: PositiveFloat


@dataclass(frozen=True)
class AngleParam(Angle):
    def __eq__(self, other):
        return self is other


@dataclass(frozen=True)
class AngleSum(Angle):
    left: Angle
    right: Angle


@dataclass(frozen=True)
class AngleDifference(Angle):
    left: Angle
    right: Angle


@dataclass(frozen=True)
class AngleBisection(Angle):
    angle: Angle


@dataclass(frozen=True)
class VectorDirection(Angle):
    vector: Vector


@dataclass(frozen=True)
class PerpendicularAngle(Angle):
    angle: Angle


@dataclass(frozen=True)
class OppositeAngle(Angle):
    angle: Angle


@dataclass(frozen=True)
class VectorPointSum(Point):
    point: Point
    vector: Vector


@dataclass(frozen=True)
class VectorPointDifference(Point):
    point: Point
    vector: Vector


@dataclass(frozen=True)
class VectorOriginSum(Point):
    vector: Vector


@dataclass(frozen=True)
class CartesianVector(Vector):
    x: float
    y: float


@dataclass(frozen=True)
class PolarVector(Vector):
    angle: Angle
    radius: Distance


@dataclass(frozen=True)
class VectorFromPoints(Vector):
    start: Point
    end: Point


@dataclass(frozen=True)
class VectorFromPoint(Vector):
    point: Point


@dataclass(frozen=True)
class VectorSum(Vector):
    left: Vector
    right: Vector


@dataclass(frozen=True)
class VectorDifference(Vector):
    left: Vector
    right: Vector


@dataclass(frozen=True)
class Line(Edge):
    pass


@dataclass(frozen=True)
class ArcBulge(Edge):
    bulge: float


@dataclass(frozen=True)
class ArcTangentStart(Edge):
    angle: Angle


@dataclass(frozen=True)
class ArcTangentEnd(Edge):
    angle: Angle


@dataclass(frozen=True)
class ArcWithSmoothStart(Edge):
    pass


@dataclass(frozen=True)
class ArcWithSmoothEnd(Edge):
    pass


@dataclass(frozen=True)
class Biarc(Edge):
    start_angle: Angle
    end_angle: Angle
    param: float


@dataclass(frozen=True)
class BiarcWithSmoothStart(Edge):
    end_angle: Angle
    param: float


@dataclass(frozen=True)
class BiarcWithSmoothEnd(Edge):
    start_angle: Angle
    param: float


@dataclass(frozen=True)
class BiarcWithSmoothExtremities(Edge):
    param: float


@dataclass(frozen=True)
class PathStart(Path):
    point: Point


@dataclass(frozen=True)
class PathEdge(Path):
    path: Path
    edge: Edge
    point: Point


@dataclass(frozen=True)
class PathClose(Shape):
    path: Path
    edge: Edge


@dataclass(frozen=True)
class ConstraintOnDistance(Constraint):
    distance: Distance
    value: PositiveFloat
    tolerance: PositiveFloat


@dataclass(frozen=True)
class ConstraintOnAngle(Constraint):
    angle: Angle
    degrees: PositiveFloat
    tolerance: PositiveFloat


@dataclass(frozen=True)
class ConstraintOnPointCoincidence(Constraint):
    first_point: Point
    second_point: Point
    tolerance: PositiveFloat
