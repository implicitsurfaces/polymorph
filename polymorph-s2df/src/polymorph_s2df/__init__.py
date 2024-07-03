from jax.numpy import isscalar
from polymorph_num import ops
from polymorph_num.expr import Expr, as_expr
from polymorph_num.vec import Vec2

from .operations import Intersection as Intersection
from .operations import Shape as Shape
from .operations import SmoothIntersection as SmoothIntersection
from .operations import SmoothUnion as SmoothUnion
from .operations import Union as Union
from .paths import (
    ArcSegment,
    ClosedPath,
    InversedSegment,
    LineSegment,
    PathSegment,
    TranslatedSegment,
)
from .shapes import BottomHalfPlane as BottomHalfPlane
from .shapes import Box as Box
from .shapes import Circle as Circle
from .shapes import LeftHalfPlane as LeftHalfPlane
from .shapes import RightHalfPlane as RightHalfPlane
from .shapes import TopHalfPlane as TopHalfPlane


def center_and_point_circle(center: Vec2, point: Vec2):
    radius = (center - point).norm()
    return Circle(radius).translate(center)


def two_corners_rectangle(corner1: Vec2, corner2: Vec2):
    size = corner1 - corner2
    return Box(size.x.abs(), size.y.abs()).translate((corner1 + corner2) / 2)


def polygon(vertices: list[Vec2]):
    segments = [
        LineSegment(vertices[i], vertices[(i + 1) % len(vertices)])
        for i in range(len(vertices))
    ]
    return ClosedPath(segments)


def bulge_arc(point1: Vec2, point2: Vec2, bulge: float):
    half_chord = (point2 - point1).norm() / 2

    # the sagitta is the perpendicular distance from the midpoint of the chord to the arc
    sagitta: Expr = as_expr(bulge).abs() * half_chord

    midpoint = (point1 + point2) / 2

    radius = (half_chord * half_chord + sagitta * sagitta) / (sagitta * 2)

    # Calculate the direction vector perpendicular to the chord
    direction = point2 - point1
    direction = direction / direction.norm()

    if bulge > 0:
        center = midpoint + direction.scale(radius - sagitta)
    else:
        center = midpoint - direction.scale(radius - sagitta)

    # Calculate the angles
    diff1 = point1 - center
    angle1 = ops.atan2(diff1.y, diff1.x)
    diff2 = point1 - center
    angle2 = ops.atan2(diff2.y, diff2.x)

    segment = TranslatedSegment(ArcSegment(angle1, angle2, radius), center)
    return segment if bulge < 0 else InversedSegment(segment)


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
