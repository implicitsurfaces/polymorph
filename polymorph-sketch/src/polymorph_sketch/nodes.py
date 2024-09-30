from dataclasses import dataclass
from typing import TypeAlias

PositiveFloat: TypeAlias = float
Sign: TypeAlias = int


@dataclass(frozen=True)
class Node:
    pass


@dataclass(frozen=True)
class Distance(Node):
    def __add__(self, other: "Distance"):
        return DistanceSum(self, other)

    def __mul__(self, other: PositiveFloat):
        return DistanceScaled(self, other)

    def __rmul__(self, other: PositiveFloat):
        return DistanceScaled(self, other)


@dataclass(frozen=True)
class Point(Node):
    def __add__(self, other: "Point"):
        return VectorSum(self, other)

    def __sub__(self, other: "Point"):
        return VectorDifference(self, other)

    def polar_angle(self):
        return PolarAngle(self)

    def polar_radius(self):
        return PolarRadius(self)


@dataclass(frozen=True)
class Angle(Node):
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
class Edge(Node):
    pass


@dataclass(frozen=True)
class Path(Node):
    pass


@dataclass(frozen=True)
class Shape(Node):
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
    point: Point


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
    point: Point


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
class VectorSum(Point):
    left: Point
    right: Point


@dataclass(frozen=True)
class VectorDifference(Point):
    left: Point
    right: Point


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
class ArcCenter(Edge):
    center: Point


@dataclass(frozen=True)
class Biarc(Edge):
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
