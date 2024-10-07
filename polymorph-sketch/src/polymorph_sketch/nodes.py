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
class VecLike:
    pass


@dataclass(frozen=True)
class Point(VecLike):
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
class Vector(VecLike):
    def __add__(self, other: "Vector") -> "Vector":
        return VectorSum(self, other)

    def __radd__(self, other: "Vector"):
        return self.__add__(other)

    def __sub__(self, other: "Vector"):
        return VectorDifference(self, other)

    def polar_angle(self) -> "Angle":
        return PolarAngle(self)

    def polar_radius(self) -> "Distance":
        return PolarRadius(self)


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
class DistanceLiteral(Distance):
    length: PositiveFloat


@dataclass(frozen=True)
class DistanceParam(Distance):
    pass


@dataclass(frozen=True)
class DistanceSum(Distance):
    left: Distance
    right: Distance


@dataclass(frozen=True)
class DistanceScaled(Distance):
    distance: Distance
    scale: PositiveFloat


@dataclass(frozen=True)
class PolarRadius(Distance):
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
    pass


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
class PolarAngle(Angle):
    vector: Vector


@dataclass(frozen=True)
class PerpendicularAngle(Angle):
    angle: Angle


@dataclass(frozen=True)
class OppositeAngle(Angle):
    angle: Angle


@dataclass(frozen=True)
class CartesianPoint(Point):
    x: float
    y: float


@dataclass(frozen=True)
class PolarPoint(Point):
    angle: Angle
    radius: Distance


@dataclass(frozen=True)
class VectorPointSum(Point):
    point: Point
    vector: Vector


@dataclass(frozen=True)
class VectorPointDifference(Point):
    point: Point
    vector: Vector


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
