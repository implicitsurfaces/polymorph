from jax.numpy import isscalar
from polymorph_num import ops
from polymorph_num.expr import Expr, Num, as_expr
from polymorph_num.vec import ValVec, Vec2, as_vec2

from .embed import EmbeddedShape as EmbeddedShape
from .embed import ModulatedExtrusion
from .operations import Intersection as Intersection
from .operations import Shape as Shape
from .operations import SmoothIntersection as SmoothIntersection
from .operations import SmoothUnion as SmoothUnion
from .operations import Union as Union
from .paths import (
    ArcSegment,
    ClosedPath,
    LineSegment,
    PathSegment,
    TranslatedSegment,
)
from .plane import XY_PLANE as XY_PLANE
from .plane import XZ_PLANE as XZ_PLANE
from .plane import YZ_PLANE as YZ_PLANE
from .plane import Plane
from .shapes import BottomHalfPlane as BottomHalfPlane
from .shapes import Box as Box
from .shapes import Circle as Circle
from .shapes import LeftHalfPlane as LeftHalfPlane
from .shapes import RightHalfPlane as RightHalfPlane
from .shapes import TopHalfPlane as TopHalfPlane
from .solids import Sphere as Sphere


def embed_in_3d(shape: Shape, plane: Plane = XY_PLANE):
    return EmbeddedShape(shape, plane)


def generic_extrusion(modulationFunction, depth: Num, base: Plane = XY_PLANE):
    return ModulatedExtrusion(modulationFunction, base, depth)


def center_and_point_circle(center: ValVec, point: ValVec):
    center = as_vec2(center)
    point = as_vec2(point)

    radius = (center - point).norm()
    return Circle(radius).translate(center)


def two_corners_rectangle(corner1: ValVec, corner2: ValVec):
    corner1 = as_vec2(corner1)
    corner2 = as_vec2(corner2)

    size = corner1 - corner2
    return Box(size.x.abs(), size.y.abs()).translate((corner1 + corner2) / 2)


def polygon(vertices: list[ValVec]):
    segments = [
        LineSegment(vertices[i], vertices[(i + 1) % len(vertices)])
        for i in range(len(vertices))
    ]
    return ClosedPath(segments)


def bulge_arc(point1: ValVec, point2: ValVec, bulge: Num):
    point1 = as_vec2(point1)
    point2 = as_vec2(point2)
    bulge = as_expr(bulge)

    half_chord = (point2 - point1).norm() / 2

    # the sagitta is the perpendicular distance from the midpoint of the chord to the arc
    sagitta: Expr = bulge.abs() * half_chord

    midpoint = (point1 + point2) / 2

    radius = (half_chord * half_chord + sagitta * sagitta) / (sagitta * 2)

    # Calculate the direction vector perpendicular to the chord
    direction = Vec2(point2.y - point1.y, point1.x - point2.x)
    direction = direction / direction.norm()

    center = midpoint + direction.scale((radius - sagitta) * bulge.sign())

    # Calculate the angles
    diff1 = point1 - center
    angle1 = ops.atan2(diff1.y, diff1.x)
    diff2 = point2 - center
    angle2 = ops.atan2(diff2.y, diff2.x)

    return TranslatedSegment(ArcSegment(angle1, angle2, radius, -bulge.sign()), center)


class DrawingPen:
    def __init__(self, start_point: tuple[float, float] = (0, 0)):
        self.current_point = start_point
        self.first_point = start_point
        self.segments: list[PathSegment] = []

    def move_to(self, point: tuple[float, float]):
        if len(self.segments) > 0:
            raise ValueError("Cannot move to a new point after drawing a segment")
        self.current_point = point
        self.first_point = point
        return self

    def line_to(self, point: tuple[float, float]):
        self.segments.append(LineSegment(Vec2(*self.current_point), Vec2(*point)))
        self.current_point = point
        return self

    def line(self, x: float, y: float):
        x0, y0 = self.current_point
        return self.line_to((x + x0, y + y0))

    def horizontal_line(self, x: float):
        return self.line(x, 0)

    def vertical_line(self, y: float):
        return self.line(0, y)

    def arc_to(self, point: tuple[float, float], bulge: float):
        self.segments.append(bulge_arc(Vec2(*self.current_point), Vec2(*point), bulge))
        self.current_point = point
        return self

    def arc(self, x: float, y: float, bulge: float):
        x0, y0 = self.current_point
        return self.arc_to((x + x0, y + y0), bulge)

    def close(self) -> ClosedPath:
        if len(self.segments) == 0:
            raise ValueError("Cannot close an empty path")
        if self.current_point != self.first_point:
            self.segments.append(
                LineSegment(Vec2(*self.current_point), Vec2(*self.first_point))
            )
        return ClosedPath(self.segments)


def draw(origin: tuple[float, float] = (0, 0)):
    return DrawingPen(origin)


def bulging_polygon(points):
    previous_point = points[0]
    previous_bulge = 0

    segments = []

    for point_or_bulge in points[1:] + [points[0]]:
        if isscalar(point_or_bulge):
            previous_bulge = point_or_bulge
        else:
            segment = (
                bulge_arc(previous_point, point_or_bulge, previous_bulge)
                if previous_bulge != 0
                else LineSegment(previous_point, point_or_bulge)
            )
            segments.append(segment)
            previous_point = point_or_bulge
            previous_bulge = 0

    return ClosedPath(segments)
