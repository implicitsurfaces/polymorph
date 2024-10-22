from typing import Callable, TypeGuard
from typing import Union as UnionType

from polymorph_app.sketch import Centroid, Constraint
from polymorph_num import ops
from polymorph_num.angle import (
    HALF_TURN,
    angle_from_deg,
    angle_from_sin,
    polar_angle_from_vec,
)
from polymorph_num.angle import (
    Angle as AngleExpr,
)
from polymorph_num.expr import Expr, as_expr
from polymorph_num.vec import Vec2
from polymorph_s2df import Circle, geometric_properties
from polymorph_s2df import Shape as ShapeExpr
from polymorph_s2df.geom_helpers import (
    biarc,
    bulging_segment_from_end_tangent,
    bulging_segment_from_start_tangent,
    cubic_biarc,
    fillet_arc_arc,
    fillet_arc_line,
    fillet_line_arc,
    fillet_line_line,
    no_inflexion_biarc,
    quadratic_biarc,
    three_point_bulging_segment,
)
from polymorph_s2df.paths import BulgingSegment, ClosedPath, LineSegment, PathSegment

from .memoizer import Memoizer
from .nodes import (
    Angle,
    AngleBisection,
    AngleDifference,
    AngleLiteral,
    AngleParam,
    AngleSum,
    ArcBulge,
    ArcEndControlPoint,
    ArcLength,
    ArcStartControlPoint,
    ArcTangentEnd,
    ArcTangentStart,
    ArcWithSmoothEnd,
    ArcWithSmoothStart,
    Biarc,
    BiarcWithSmoothEnd,
    BiarcWithSmoothExtremities,
    BiarcWithSmoothStart,
    CartesianVector,
    ConstraintOnAngle,
    ConstraintOnDistance,
    ConstraintOnPointCoincidence,
    ConstraintOnShapeBoundary,
    CubicBiarc,
    Distance,
    DistanceLiteral,
    DistanceParam,
    DistanceScaled,
    DistanceSum,
    Edge,
    LeftBiarc,
    Line,
    OppositeAngle,
    Path,
    PathClose,
    PathEdge,
    PathStart,
    PerpendicularAngle,
    Point,
    PolarVector,
    PositiveFloat,
    Q1Angle,
    Q1AngleParam,
    QuadraticBiarc,
    RealParam,
    RealValue,
    RightBiarc,
    Shape,
    ShapeCircle,
    ShapeDifference,
    ShapeIntersection,
    ShapeMorph,
    ShapeRotation,
    ShapeScale,
    ShapeShell,
    ShapeTranslation,
    ShapeUnion,
    ThreePointArc,
    Vector,
    VectorDifference,
    VectorDirection,
    VectorFromPoint,
    VectorFromPoints,
    VectorNorm,
    VectorOriginSum,
    VectorPointDifference,
    VectorPointSum,
    VectorSum,
)

memoizer = Memoizer()


def is_positive_float(x: float) -> PositiveFloat:
    if x < 0:
        raise ValueError(f"Expected positive float, got {x}")
    return x


@memoizer.memoize()
def sketch_real_value(node: int | float | RealValue) -> Expr:
    if isinstance(node, Distance):
        return sketch_distance(node)
    match node:
        case RealParam():
            return ops.param()
        case float(value):
            return as_expr(value)
        case int(value):
            return as_expr(value)
        case _:
            raise ValueError(f"Unexpected real value node: {node}")


@memoizer.memoize()
def sketch_distance(node: Distance) -> Expr:
    match node:
        case DistanceLiteral(length):
            return as_expr(length)
        case DistanceParam():
            p = ops.param()
            return p * p
        case DistanceSum(left, right):
            return sketch_distance(left) + sketch_distance(right)
        case DistanceScaled(distance, scale):
            return scale * sketch_distance(distance)
        case VectorNorm(vector):
            p = sketch_vector(vector)
            return p.norm()
        case ArcLength(angle, radius):
            a = sketch_angle(angle)
            r = sketch_distance(radius)
            return a.as_rad() * r
        case _:
            raise ValueError(f"Unexpected distance node: {node}")


@memoizer.memoize()
def sketch_angle(node: Angle) -> AngleExpr:
    match node:
        case AngleLiteral(degrees):
            return angle_from_deg(as_expr(is_positive_float(degrees)))
        case AngleParam():
            return angle_from_sin(2 * ops.param().sigmoid() - 1).double()
        case AngleSum(left, right):
            return sketch_angle(left) + sketch_angle(right)
        case AngleDifference(left, right):
            return sketch_angle(left) - sketch_angle(right)
        case AngleBisection(angle):
            return sketch_angle(angle).half()
        case VectorDirection(vector):
            p = sketch_vector(vector)
            return polar_angle_from_vec(p)
        case PerpendicularAngle(angle):
            return sketch_angle(angle).perp()
        case OppositeAngle(angle):
            return HALF_TURN + sketch_angle(angle)
        case Q1Angle(base):
            a = sketch_angle(base)
            return AngleExpr(a.cos().abs(), a.sin().abs())
        case Q1AngleParam():
            return angle_from_sin(ops.param().sigmoid())

        case _:
            raise ValueError(f"Unexpected angle node: {node}")


@memoizer.memoize()
def sketch_point(node: Point) -> Vec2:
    match node:
        case VectorOriginSum(vector):
            return sketch_vector(vector)
        case VectorPointSum(point, vector):
            return sketch_point(point) + sketch_vector(vector)
        case VectorPointDifference(point, vector):
            return sketch_point(point) - sketch_vector(vector)
        case Centroid(shape):
            shape = sketch_shape(shape)
            return geometric_properties.centroid(shape)
        case _:
            raise ValueError(f"Unexpected point node: {node}")


@memoizer.memoize()
def sketch_vector(node: Vector) -> Vec2:
    match node:
        case CartesianVector(x, y):
            return Vec2(sketch_real_value(x), sketch_real_value(y))
        case PolarVector(angle, distance):
            return sketch_angle(angle).as_vec().scale(sketch_distance(distance))
        case VectorFromPoint(point):
            return sketch_point(point)
        case VectorFromPoints(start, end):
            return sketch_point(end) - sketch_point(start)
        case VectorSum(left, right):
            return sketch_vector(left) + sketch_vector(right)
        case VectorDifference(left, right):
            return sketch_vector(left) - sketch_vector(right)
        case _:
            raise ValueError(f"Unexpected vector node: {node}")


class BiarcSegment:
    def __init__(self, start, end, start_tangent, end_tangent, param):
        self.subsegments = biarc(
            start.x,
            start.y,
            start_tangent,
            end.x,
            end.y,
            end_tangent,
            param,
        )

    def start_tangent(self):
        return self.subsegments[0].start_tangent()

    def end_tangent(self):
        return self.subsegments[-1].end_tangent()

    @property
    def end(self):
        return self.subsegments[-1].end


class RightBiarcSegment(BiarcSegment):
    def __init__(self, start, end, start_tangent, end_tangent):
        chord_angle = polar_angle_from_vec(end - start)
        self.subsegments = no_inflexion_biarc(
            start.x,
            start.y,
            chord_angle - start_tangent,
            end.x,
            end.y,
            chord_angle + end_tangent,
        )


class LeftBiarcSegment(BiarcSegment):
    def __init__(self, start, end, start_tangent, end_tangent):
        chord_angle = polar_angle_from_vec(end - start)
        self.subsegments = no_inflexion_biarc(
            start.x,
            start.y,
            chord_angle + start_tangent,
            end.x,
            end.y,
            chord_angle - end_tangent,
        )


class QuadraticBiarcSegment(BiarcSegment):
    def __init__(self, start, end, control):
        self.subsegments = quadratic_biarc(
            start.x,
            start.y,
            end.x,
            end.y,
            control.x,
            control.y,
        )


class CubicBiarcSegment(BiarcSegment):
    def __init__(self, start, end, start_tangent, end_tangent):
        self.subsegments = cubic_biarc(
            start.x,
            start.y,
            end.x,
            end.y,
            start_tangent.x,
            start_tangent.y,
            end_tangent.x,
            end_tangent.y,
        )


class IncompleteEdge:
    start: Vec2
    end: Vec2

    def fix(
        self, i, edges: list[UnionType["IncompleteEdge", PathSegment, BiarcSegment]]
    ):
        raise NotImplementedError


class IncompleteArcSmoothStart(IncompleteEdge):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def fix(self, i, edges: list[IncompleteEdge | PathSegment | BiarcSegment]):
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

    def fix(self, i, edges: list[IncompleteEdge | PathSegment | BiarcSegment]):
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

    def fix(self, i, edges: list[IncompleteEdge | PathSegment | BiarcSegment]):
        previous = edges[i - 1]
        if isinstance(previous, IncompleteEdge):
            return self

        return BiarcSegment(
            self.start, self.end, previous.end_tangent(), self.end_tangent, self.param
        )


class IncompleteBiArcSmoothEnd(IncompleteEdge):
    def __init__(self, start, end, start_tangent, param):
        self.start = start
        self.end = end
        self.start_tangent = start_tangent
        self.param = param

    def fix(self, i, edges: list[IncompleteEdge | PathSegment | BiarcSegment]):
        next = edges[(i + 1) % len(edges)]
        if isinstance(next, IncompleteEdge):
            return self

        return BiarcSegment(
            self.start, self.end, self.start_tangent, next.start_tangent(), self.param
        )


class IncompleteBiArcSmoothExtremities(IncompleteEdge):
    def __init__(self, start, end, param):
        self.start = start
        self.end = end
        self.param = param

    def fix(self, i, edges: list[IncompleteEdge | PathSegment | BiarcSegment]):
        next = edges[(i + 1) % len(edges)]
        if isinstance(next, IncompleteEdge):
            return self

        previous = edges[i - 1]
        if isinstance(previous, IncompleteEdge):
            return self

        return BiarcSegment(
            self.start,
            self.end,
            previous.end_tangent(),
            next.start_tangent(),
            self.param,
        )


def is_incomplete(edge) -> TypeGuard[IncompleteEdge]:
    return isinstance(edge, IncompleteEdge)


@memoizer.memoize()
def sketch_edge(
    node: Edge,
) -> Callable[[Vec2, Vec2], PathSegment | IncompleteEdge | BiarcSegment]:
    match node:
        case Line():
            return lambda p0, p1: LineSegment(p0, p1)

        case ArcBulge(bulge):
            return lambda p0, p1: BulgingSegment(p0, p1, sketch_real_value(bulge))

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
            return lambda p0, p1: BiarcSegment(
                p0,
                p1,
                sketch_angle(start_tangent),
                sketch_angle(end_tangent),
                sketch_angle(param),
            )

        case BiarcWithSmoothStart(end_tangent, param):
            return lambda p0, p1: IncompleteBiArcSmoothStart(
                p0, p1, sketch_angle(end_tangent), sketch_real_value(param)
            )

        case BiarcWithSmoothEnd(start_tangent, param):
            return lambda p0, p1: IncompleteBiArcSmoothEnd(
                p0, p1, sketch_angle(start_tangent), sketch_real_value(param)
            )

        case BiarcWithSmoothExtremities(param):
            return lambda p0, p1: IncompleteBiArcSmoothExtremities(
                p0, p1, sketch_real_value(param)
            )

        case RightBiarc(start_angle, end_angle):
            return lambda p0, p1: RightBiarcSegment(
                p0,
                p1,
                sketch_angle(start_angle),
                sketch_angle(end_angle),
            )

        case LeftBiarc(start_angle, end_angle):
            return lambda p0, p1: LeftBiarcSegment(
                p0,
                p1,
                sketch_angle(start_angle),
                sketch_angle(end_angle),
            )

        case ArcStartControlPoint(control_point):
            return lambda p0, p1: bulging_segment_from_start_tangent(
                p0, p1, polar_angle_from_vec(sketch_point(control_point) - p0)
            )

        case ArcEndControlPoint(control_point):
            return lambda p0, p1: bulging_segment_from_end_tangent(
                p0,
                p1,
                polar_angle_from_vec(sketch_point(control_point) - p1).opposite(),
            )

        case ThreePointArc(on_arc):
            return lambda p0, p1: three_point_bulging_segment(
                p0, p1, sketch_point(on_arc)
            )

        case CubicBiarc(start_control_point, end_control_point):
            return lambda p0, p1: CubicBiarcSegment(
                p0,
                p1,
                sketch_point(start_control_point),
                sketch_point(end_control_point),
            )

        case QuadraticBiarc(control_point):
            return lambda p0, p1: QuadraticBiarcSegment(
                p0,
                p1,
                polar_angle_from_vec(sketch_point(control_point) - p0),
            )
        case _:
            raise ValueError(f"Unexpected edge node: {node}")


def fillet_edges(edge1: PathSegment, edge2: PathSegment | BiarcSegment, radius: Expr):
    if isinstance(edge2, BiarcSegment):
        return fillet_edges(edge1, edge2.subsegments[0], radius) + [
            edge2.subsegments[-1]
        ]

    if isinstance(edge1, LineSegment) and isinstance(edge2, LineSegment):
        return fillet_line_line(edge1, edge2, radius)
    if isinstance(edge1, LineSegment) and isinstance(edge2, BulgingSegment):
        return fillet_line_arc(edge1, edge2, radius)
    if isinstance(edge1, BulgingSegment) and isinstance(edge2, LineSegment):
        return fillet_arc_line(edge1, edge2, radius)
    if isinstance(edge1, BulgingSegment) and isinstance(edge2, BulgingSegment):
        return fillet_arc_arc(edge1, edge2, radius)

    raise ValueError(f"Unexpected edge combination: {edge1} and {edge2}")


class PathBuilder:
    def __init__(self, first_point: Vec2):
        self.first_point = first_point
        self._edges: list[IncompleteEdge | PathSegment | BiarcSegment] = []
        self._corner_radiuses: dict[int, Expr] = {}

    def register_corner_radius(self, radius):
        i = len(self._edges)
        self._corner_radiuses[i] = radius

    @property
    def current_point(self):
        if not len(self._edges):
            return self.first_point
        return self._edges[-1].end

    def append(self, edge: PathSegment | IncompleteEdge | BiarcSegment):
        self._edges.append(edge)

    def finalize(self) -> list[PathSegment]:
        edges = self._edges
        incomplete_count = len([e for e in edges if is_incomplete(e)])

        # We complete the path by figuring out the tangents dependencies.
        while incomplete_count > 0:
            new_eges = []
            for i, e in enumerate(edges):
                if is_incomplete(e):
                    e = e.fix(i, edges)
                new_eges.append(e)
            edges = new_eges

            new_incomplete_count = len([e for e in edges if is_incomplete(e)])
            if new_incomplete_count == incomplete_count:
                raise ValueError("Failed to finalize path - incomplete edges remain")
            incomplete_count = new_incomplete_count

        # We now have a list of segments, we need to add the corner radiuses (and expand biarc segments)
        final_edges: list[PathSegment] = []

        for i, edge in enumerate(edges):
            if isinstance(edge, IncompleteEdge):
                raise ValueError("Incomplete edges should have been fixed by now")
            elif i in self._corner_radiuses and i != 0:
                final_edges.extend(
                    fillet_edges(final_edges.pop(), edge, self._corner_radiuses[i])
                )
            elif isinstance(edge, BiarcSegment):
                final_edges.extend(edge.subsegments)
            else:
                final_edges.append(edge)

        if 0 in self._corner_radiuses:
            final_edges.extend(
                fillet_edges(
                    final_edges.pop(-1), final_edges.pop(0), self._corner_radiuses[0]
                )
            )

        return final_edges


def sketch_path(node: Path) -> PathBuilder:
    match node:
        case PathStart(point, corner_radius):
            p = sketch_point(point)
            builder = PathBuilder(p)
            if corner_radius is not None:
                builder.register_corner_radius(sketch_distance(corner_radius))
            return builder

        case PathEdge(path, edge, point, corner_radius):
            edges_list = sketch_path(path)
            p = sketch_point(point)
            e = sketch_edge(edge)(edges_list.current_point, p)
            edges_list.append(e)
            if corner_radius is not None:
                edges_list.register_corner_radius(sketch_distance(corner_radius))
            return edges_list
        case _:
            raise ValueError(f"Unexpected path node: {node}")


@memoizer.memoize()
def sketch_shape(node: Shape) -> ShapeExpr:
    match node:
        case PathClose(path, edge):
            edges_list = sketch_path(path)
            first_point = edges_list.first_point
            last_point = edges_list.current_point

            last_edge = sketch_edge(edge)(last_point, first_point)
            edges_list.append(last_edge)
            return ClosedPath(edges_list.finalize())

        case ShapeTranslation(shape, vector):
            return sketch_shape(shape).translate(sketch_vector(vector))

        case ShapeRotation(shape, angle):
            return sketch_shape(shape).rotate(sketch_angle(angle).as_rad())

        case ShapeUnion(a, b):
            return sketch_shape(a).union(sketch_shape(b))

        case ShapeIntersection(a, b):
            return sketch_shape(a).intersect(sketch_shape(b))

        case ShapeDifference(a, b):
            return sketch_shape(a).substract(sketch_shape(b))

        case ShapeShell(shape, thickness):
            return sketch_shape(shape).shell(sketch_distance(thickness))

        case ShapeScale(shape, factor):
            return sketch_shape(shape).scale(sketch_distance(factor))

        case ShapeMorph(a, b, t):
            return sketch_shape(a).morph(sketch_real_value(t), sketch_shape(b))

        case ShapeCircle(radius, center):
            return Circle(sketch_distance(radius)).translate(sketch_point(center))

        case _:
            raise ValueError(f"Unexpected path node: {node}")


@memoizer.memoize()
def constraint_loss(node: Constraint) -> Expr:
    match node:
        case ConstraintOnDistance(distance, value, tolerance):
            dist = sketch_distance(distance)
            target = as_expr(is_positive_float(value))

            tol = as_expr(is_positive_float(tolerance))

            # We want to minimize the difference between the distance and the target,
            # but we want to weight the difference by the tolerance (so that the tolerance
            # corresponds to the standard deviation of a normal distribution)
            #
            # I apply the scaling by the tolerance before the subtraction, I *think*
            # that it might be better for numerical stability, but I need to learn more
            # about this.
            weighted_diff = (dist / tol) - (target / tol)

            return weighted_diff * weighted_diff

        case ConstraintOnAngle(angle, degrees, tolerance):
            theta = sketch_angle(angle)
            target = angle_from_deg(as_expr(is_positive_float(degrees)))
            tol = as_expr(is_positive_float(tolerance))

            loss = 2.0 - ((theta - target).cos() + 1)
            return loss / (tol * 2)

        case ConstraintOnPointCoincidence(first_point, second_point, tolerance):
            p1 = sketch_point(first_point)
            p2 = sketch_point(second_point)
            tol = as_expr(is_positive_float(tolerance))

            error_vec = (p1 / tol) - (p2 / tol)
            return error_vec.x * error_vec.x + error_vec.y * error_vec.y

        case ConstraintOnShapeBoundary(shape, point):
            p = sketch_point(point)
            d = sketch_shape(shape).distance(p.x, p.y)
            return d * d

        case _:
            raise ValueError(f"Unexpected constraint node: {node}")


def reset_cache():
    memoizer.reset_all_caches()


def debug_inner_node(node):
    if isinstance(node, Angle):
        return sketch_angle(node).as_deg()
    if isinstance(node, Distance):
        return sketch_distance(node)
    if isinstance(node, Point):
        return sketch_point(node)
    if isinstance(node, Vector):
        return sketch_vector(node)

    raise ValueError(f"Unexpected node: {node}")
