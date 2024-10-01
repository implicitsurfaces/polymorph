from typing import TypeGuard

from polymorph_num.angle import HALF_TURN, angle_from_deg, polar_angle_from_vec
from polymorph_num.expr import as_expr
from polymorph_num.ops import param
from polymorph_num.vec import Vec2
from polymorph_s2df.geom_helpers import (
    bulging_segment_from_center,
    bulging_segment_from_end_tangent,
    bulging_segment_from_start_tangent,
)
from polymorph_s2df.paths import BulgingSegment, ClosedPath, LineSegment, PathSegment

from .nodes import (
    AngleBisection,
    AngleDifference,
    AngleLiteral,
    AngleParam,
    AngleSum,
    ArcBulge,
    ArcCenter,
    ArcTangentEnd,
    ArcTangentStart,
    ArcWithSmoothEnd,
    ArcWithSmoothStart,
    CartesianPoint,
    DistanceLiteral,
    DistanceParam,
    DistanceScaled,
    DistanceSum,
    Line,
    Node,
    OppositeAngle,
    PathClose,
    PathEdge,
    PathStart,
    PerpendicularAngle,
    PolarAngle,
    PolarPoint,
    PolarRadius,
    PositiveFloat,
    VectorDifference,
    VectorSum,
)


def is_positive_float(x: float) -> PositiveFloat:
    if x <= 0:
        raise ValueError(f"Expected positive float, got {x}")
    return x


class IncompleteEdge:
    start: Vec2
    end: Vec2

    def fix(self, i, edges):
        raise NotImplementedError


class IncompleteArcSmoothStart(IncompleteEdge):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def fix(self, i, edges: list[IncompleteEdge | PathSegment]):
        previous = edges[i - 1]
        if isinstance(previous, IncompleteEdge):
            return self

        return bulging_segment_from_start_tangent(
            self.start, self.end, previous.end_tangent()
        )


class IncompleteArcSmoothEnd(IncompleteEdge):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def fix(self, i, edges: list[IncompleteEdge | PathSegment]):
        next = edges[(i + 1) % len(edges)]
        if isinstance(next, IncompleteEdge):
            return self

        return bulging_segment_from_end_tangent(
            self.start, self.end, next.start_tangent()
        )


def is_incomplete(edge) -> TypeGuard[IncompleteEdge]:
    return isinstance(edge, IncompleteEdge)


class EdgeList:
    def __init__(self, first_point: Vec2):
        self.first_point = first_point
        self._edges: list[IncompleteEdge | PathSegment] = []

    @property
    def current_point(self):
        if not len(self._edges):
            return self.first_point
        return self._edges[-1].end

    def append(self, edge: PathSegment | IncompleteEdge):
        self._edges.append(edge)

    def finalize(self):
        edges = self._edges
        incomplete_count = len([e for e in edges if is_incomplete(e)])

        while incomplete_count > 0:
            edges = [
                e.fix(i, edges) if is_incomplete(e) else e for i, e in enumerate(edges)
            ]
            new_incomplete_count = len([e for e in edges if is_incomplete(e)])
            if new_incomplete_count == incomplete_count:
                raise ValueError("Failed to finalize path - incomplete edges remain")
            incomplete_count = new_incomplete_count

        return edges


def sketch(node: Node):
    match node:
        case DistanceLiteral(length):
            return as_expr(length)
        case DistanceParam():
            return param()
        case DistanceSum(left, right):
            return sketch(left) + sketch(right)
        case DistanceScaled(distance, scale):
            return scale * sketch(distance)
        case PolarRadius(point):
            p = sketch(point)
            return p.norm()
        case AngleLiteral(degrees):
            return angle_from_deg(as_expr(is_positive_float(degrees)))
        case AngleParam():
            return param()
        case AngleSum(left, right):
            return sketch(left) + sketch(right)
        case AngleDifference(left, right):
            return sketch(left) - sketch(right)
        case AngleBisection(angle):
            return sketch(angle).half()
        case PolarAngle(point):
            p = sketch(point)
            return polar_angle_from_vec(p)
        case PerpendicularAngle(angle):
            return sketch(angle).perp()
        case OppositeAngle(angle):
            return HALF_TURN + sketch(angle)
        case CartesianPoint(x, y):
            return Vec2(x, y)
        case PolarPoint(angle, radius):
            r = sketch(radius)
            a = sketch(angle)
            return a.as_vec().scale(r)
        case VectorSum(left, right):
            return sketch(left) + sketch(right)
        case VectorDifference(left, right):
            return sketch(left) - sketch(right)
        case PathClose(path, edge):
            edges_list = sketch(path)
            first_point = edges_list.first_point
            last_point = edges_list.current_point

            last_edge = sketch(edge)(last_point, first_point)
            edges_list.append(last_edge)
            return ClosedPath(edges_list.finalize())
        case PathStart(point):
            p = sketch(point)
            return EdgeList(p)
        case PathEdge(path, point, edge):
            edges_list = sketch(path)
            p = sketch(point)
            e = sketch(edge)(edges_list.current_point, p)

            edges_list.append(e)
            return edges_list
        case Line():
            return lambda p0, p1: LineSegment(p0, p1)

        case ArcBulge(bulge):
            return lambda p0, p1: BulgingSegment(p0, p1, bulge)

        case ArcTangentStart(angle):
            return lambda p0, p1: bulging_segment_from_start_tangent(
                p0, p1, sketch(angle)
            )

        case ArcTangentEnd(angle):
            return lambda p0, p1: bulging_segment_from_end_tangent(
                p0, p1, sketch(angle)
            )

        case ArcWithSmoothStart():
            return IncompleteArcSmoothStart

        case ArcWithSmoothEnd():
            return IncompleteArcSmoothEnd

        case ArcCenter(center):
            c = sketch(center)
            return lambda p0, p1: bulging_segment_from_center(p0, p1, c)
