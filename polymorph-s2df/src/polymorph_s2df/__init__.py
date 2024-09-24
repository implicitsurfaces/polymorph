from jax.numpy import isscalar
from polymorph_num import ops as ops
from polymorph_num.angle import NO_TURN, angle_from_rad
from polymorph_num.expr import Expr as Expr
from polymorph_num.expr import Num, as_expr
from polymorph_num.vec import ValVec, Vec2, as_vec2

from polymorph_s2df.geom_helpers import (
    biarc,
    bulging_segment_from_start_tangent,
)

from .embed import EmbeddedShape as EmbeddedShape
from .embed import ModulatedExtrusion, SweepWand
from .operations import Intersection as Intersection
from .operations import Shape as Shape
from .operations import SmoothIntersection as SmoothIntersection
from .operations import SmoothUnion as SmoothUnion
from .operations import Union as Union
from .paths import (
    BulgingSegment as BulgingSegment,
)
from .paths import (
    ClosedPath,
    LineSegment,
    PathSegment,
)
from .paths import (
    TranslatedSegment as TranslatedSegment,
)
from .plane import X_AXIS as X_AXIS
from .plane import XY_PLANE as XY_PLANE
from .plane import XZ_PLANE as XZ_PLANE
from .plane import Y_AXIS as Y_AXIS
from .plane import YZ_PLANE as YZ_PLANE
from .plane import Z_AXIS as Z_AXIS
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


class DrawingPen:
    current_point: tuple[float, float]
    first_point: tuple[float, float]
    segments: list[PathSegment]

    def __init__(self, start_point: tuple[float, float] = (0, 0)):
        self.current_point = start_point
        self.first_point = start_point
        self.segments: list[PathSegment] = []

    def _current_angle(self):
        if len(self.segments) == 0:
            return NO_TURN
        last_segment = self.segments[-1]
        return last_segment.end_tangent()

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
        self.segments.append(
            BulgingSegment(Vec2(*self.current_point), Vec2(*point), bulge)
        )
        self.current_point = point
        return self

    def arc(self, x: float, y: float, bulge: float):
        x0, y0 = self.current_point
        return self.arc_to((x + x0, y + y0), bulge)

    def tangent_arc_to(
        self,
        point: tuple[float, float],
        angle: float | None = None,
    ):
        first = as_vec2(self.current_point)
        second = as_vec2(point)

        self.segments.append(
            bulging_segment_from_start_tangent(
                first,
                second,
                self._current_angle()
                if angle is None
                else angle_from_rad(as_expr(angle)),
            )
        )
        self.current_point = point
        return self

    def tangent_arc(self, x: float, y: float, angle: float | None = None):
        x0, y0 = self.current_point
        return self.tangent_arc_to((x + x0, y + y0), angle)

    def biarc_to(
        self,
        end: tuple[float, float],
        angle: float,
        start_angle: float | None = None,
        param: float = 0.5,
    ):
        x0, y0 = self.current_point
        x1, y1 = end

        self.segments.extend(
            biarc(
                as_expr(x0),
                as_expr(y0),
                self._current_angle()
                if start_angle is None
                else angle_from_rad(as_expr(start_angle)),
                as_expr(x1),
                as_expr(y1),
                angle_from_rad(as_expr(angle)),
                as_expr(param),
            )
        )
        self.current_point = (x1, y1)
        return self

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
                BulgingSegment(previous_point, point_or_bulge, previous_bulge)
                if previous_bulge != 0
                else LineSegment(previous_point, point_or_bulge)
            )
            segments.append(segment)
            previous_point = point_or_bulge
            previous_bulge = 0

    return ClosedPath(segments)


def sweep(shape: Shape, plane: Plane = XY_PLANE):
    return SweepWand(shape, plane)
