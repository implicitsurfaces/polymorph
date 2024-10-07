import collections.abc
from typing import Callable, Sequence, TypeGuard, Union

from polymorph_num import ops
from polymorph_num.angle import (
    HALF_TURN,
    angle_from_deg,
    angle_from_rad,
    polar_angle_from_vec,
)
from polymorph_num.angle import (
    Angle as AngleExpr,
)
from polymorph_num.expr import Expr, as_expr
from polymorph_num.vec import Vec2
from polymorph_s2df import Shape as ShapeExpr
from polymorph_s2df.geom_helpers import (
    biarc,
    bulging_segment_from_end_tangent,
    bulging_segment_from_start_tangent,
)
from polymorph_s2df.paths import BulgingSegment, ClosedPath, LineSegment, PathSegment

from .nodes import (
    Angle,
    AngleBisection,
    AngleDifference,
    AngleLiteral,
    AngleParam,
    AngleSum,
    ArcBulge,
    ArcLength,
    ArcTangentEnd,
    ArcTangentStart,
    ArcWithSmoothEnd,
    ArcWithSmoothStart,
    Biarc,
    BiarcWithSmoothEnd,
    BiarcWithSmoothExtremities,
    BiarcWithSmoothStart,
    CartesianPoint,
    Distance,
    DistanceLiteral,
    DistanceParam,
    DistanceScaled,
    DistanceSum,
    Edge,
    Line,
    OppositeAngle,
    Path,
    PathClose,
    PathEdge,
    PathStart,
    PerpendicularAngle,
    Point,
    PolarAngle,
    PolarPoint,
    PolarRadius,
    PositiveFloat,
    Shape,
    VectorDifference,
    VectorSum,
)


def is_positive_float(x: float) -> PositiveFloat:
    if x <= 0:
        raise ValueError(f"Expected positive float, got {x}")
    return x


def sketch_distance(node: Distance) -> Expr:
    match node:
        case DistanceLiteral(length):
            return as_expr(length)
        case DistanceParam():
            return ops.param()
        case DistanceSum(left, right):
            return sketch_distance(left) + sketch_distance(right)
        case DistanceScaled(distance, scale):
            return scale * sketch_distance(distance)
        case PolarRadius(point):
            p = sketch_point(point)
            return p.norm()
        case ArcLength(angle, radius):
            a = sketch_angle(angle)
            r = sketch_distance(radius)
            return a.as_rad() * r
        case _:
            raise ValueError(f"Unexpected distance node: {node}")


def sketch_angle(node: Angle) -> AngleExpr:
    match node:
        case AngleLiteral(degrees):
            return angle_from_deg(as_expr(is_positive_float(degrees)))
        case AngleParam():
            return angle_from_rad(ops.param().atan())
        case AngleSum(left, right):
            return sketch_angle(left) + sketch_angle(right)
        case AngleDifference(left, right):
            return sketch_angle(left) - sketch_angle(right)
        case AngleBisection(angle):
            return sketch_angle(angle).half()
        case PolarAngle(point):
            p = sketch_point(point)
            return polar_angle_from_vec(p)
        case PerpendicularAngle(angle):
            return sketch_angle(angle).perp()
        case OppositeAngle(angle):
            return HALF_TURN + sketch_angle(angle)
        case _:
            raise ValueError(f"Unexpected angle node: {node}")


def sketch_point(node: Point) -> Vec2:
    match node:
        case CartesianPoint(x, y):
            return Vec2(x, y)
        case PolarPoint(angle, radius):
            r = sketch_distance(radius)
            a = sketch_angle(angle)
            return a.as_vec().scale(r)
        case VectorSum(left, right):
            return sketch_point(left) + sketch_point(right)
        case VectorDifference(left, right):
            return sketch_point(left) - sketch_point(right)
        case _:
            raise ValueError(f"Unexpected point node: {node}")


class IncompleteEdge:
    start: Vec2
    end: Vec2

    def fix(self, i, edges: list[Union["IncompleteEdge", PathSegment]]):
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


class IncompleteBiArcSmoothStart(IncompleteEdge):
    def __init__(self, start, end, end_tangent, param):
        self.start = start
        self.end = end
        self.end_tangent = end_tangent
        self.param = param

    def fix(self, i, edges: list[IncompleteEdge | PathSegment]):
        previous = edges[i - 1]
        if isinstance(previous, IncompleteEdge):
            return self

        return biarc(
            self.start.x,
            self.start.y,
            previous.end_tangent(),
            self.end.x,
            self.end.y,
            self.end_tangent,
            self.param,
        )


class IncompleteBiArcSmoothEnd(IncompleteEdge):
    def __init__(self, start, end, start_tangent, param):
        self.start = start
        self.end = end
        self.start_tangent = start_tangent
        self.param = param

    def fix(self, i, edges: list[IncompleteEdge | PathSegment]):
        next = edges[(i + 1) % len(edges)]
        if isinstance(next, IncompleteEdge):
            return self

        return biarc(
            self.start.x,
            self.start.y,
            self.start_tangent,
            self.end.x,
            self.end.y,
            next.start_tangent(),
            self.param,
        )


class IncompleteBiArcSmoothExtremities(IncompleteEdge):
    def __init__(self, start, end, param):
        self.start = start
        self.end = end
        self.param = param

    def fix(self, i, edges: list[IncompleteEdge | PathSegment]):
        next = edges[(i + 1) % len(edges)]
        if isinstance(next, IncompleteEdge):
            return self

        previous = edges[i - 1]
        if isinstance(previous, IncompleteEdge):
            return self

        return biarc(
            self.start.x,
            self.start.y,
            previous.end_tangent(),
            self.end.x,
            self.end.y,
            next.start_tangent(),
            self.param,
        )


def is_incomplete(edge) -> TypeGuard[IncompleteEdge]:
    return isinstance(edge, IncompleteEdge)


def sketch_edge(
    node: Edge,
) -> Callable[[Vec2, Vec2], PathSegment | IncompleteEdge | Sequence[PathSegment]]:
    match node:
        case Line():
            return lambda p0, p1: LineSegment(p0, p1)

        case ArcBulge(bulge):
            return lambda p0, p1: BulgingSegment(p0, p1, bulge)

        case ArcTangentStart(angle):
            return lambda p0, p1: bulging_segment_from_start_tangent(
                p0, p1, sketch_angle(angle)
            )

        case ArcTangentEnd(angle):
            return lambda p0, p1: bulging_segment_from_end_tangent(
                p0, p1, sketch_angle(angle)
            )

        case ArcWithSmoothStart():
            return IncompleteArcSmoothStart

        case ArcWithSmoothEnd():
            return IncompleteArcSmoothEnd

        case Biarc(start_tangent, end_tangent, param):
            return lambda p0, p1: biarc(
                p0.x,
                p0.y,
                sketch_angle(start_tangent),
                p1.x,
                p1.y,
                sketch_angle(end_tangent),
                as_expr(param),
            )

        case BiarcWithSmoothStart(end_tangent, param):
            return lambda p0, p1: IncompleteBiArcSmoothStart(
                p0, p1, sketch_angle(end_tangent), as_expr(param)
            )

        case BiarcWithSmoothEnd(start_tangent, param):
            return lambda p0, p1: IncompleteBiArcSmoothEnd(
                p0, p1, sketch_angle(start_tangent), as_expr(param)
            )

        case BiarcWithSmoothExtremities(param):
            return lambda p0, p1: IncompleteBiArcSmoothExtremities(
                p0, p1, as_expr(param)
            )
        case _:
            raise ValueError(f"Unexpected edge node: {node}")


class PathBuilder:
    def __init__(self, first_point: Vec2):
        self.first_point = first_point
        self._edges: list[IncompleteEdge | PathSegment] = []

    @property
    def current_point(self):
        if not len(self._edges):
            return self.first_point
        return self._edges[-1].end

    def append(self, edge: PathSegment | IncompleteEdge | Sequence[PathSegment]):
        if isinstance(edge, collections.abc.Sequence):
            self._edges.extend(edge)
        else:
            self._edges.append(edge)

    def finalize(self) -> list[PathSegment]:
        edges = self._edges
        incomplete_count = len([e for e in edges if is_incomplete(e)])

        while incomplete_count > 0:
            new_eges = []
            for i, e in enumerate(edges):
                if is_incomplete(e):
                    e = e.fix(i, edges)
                if isinstance(e, list):
                    new_eges.extend(e)
                else:
                    new_eges.append(e)
            edges = new_eges

            new_incomplete_count = len([e for e in edges if is_incomplete(e)])
            if new_incomplete_count == incomplete_count:
                raise ValueError("Failed to finalize path - incomplete edges remain")
            incomplete_count = new_incomplete_count

        return edges  # type: ignore


def sketch_path(node: Path) -> PathBuilder:
    match node:
        case PathStart(point):
            p = sketch_point(point)
            return PathBuilder(p)
        case PathEdge(path, edge, point):
            edges_list = sketch_path(path)
            p = sketch_point(point)
            e = sketch_edge(edge)(edges_list.current_point, p)
            edges_list.append(e)
            return edges_list
        case _:
            raise ValueError(f"Unexpected path node: {node}")


def sketch_shape(node: Shape) -> ShapeExpr:
    match node:
        case PathClose(path, edge):
            edges_list = sketch_path(path)
            first_point = edges_list.first_point
            last_point = edges_list.current_point

            last_edge = sketch_edge(edge)(last_point, first_point)
            edges_list.append(last_edge)
            return ClosedPath(edges_list.finalize())
        case _:
            raise ValueError(f"Unexpected path node: {node}")
