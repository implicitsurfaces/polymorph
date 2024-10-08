from polymorph_num.expr import ZERO
from polymorph_num.unit import Unit
from polymorph_s2df import Shape

from .eval import constraint_loss, sketch_shape
from .eval import debug_inner_node as debug_inner_node
from .nodes import (
    Angle,
    AngleLiteral,
    AngleParam,
    ArcBulge,
    ArcTangentEnd,
    ArcTangentStart,
    ArcWithSmoothEnd,
    ArcWithSmoothStart,
    Biarc,
    BiarcWithSmoothEnd,
    BiarcWithSmoothExtremities,
    BiarcWithSmoothStart,
    CartesianPoint,
    ConstraintOnAngle,
    ConstraintOnDistance,
    ConstraintOnPointCoincidence,
    Distance,
    DistanceLiteral,
    DistanceParam,
    Line,
    Path,
    PathClose,
    PathEdge,
    PathStart,
    Point,
    PolarPoint,
)


def distance_param():
    return DistanceParam()


def angle_param():
    return AngleParam()


def point_param():
    return PolarPoint(AngleParam(), DistanceParam())


def as_distance(distance: float | Distance) -> Distance:
    if isinstance(distance, Distance):
        return distance
    return DistanceLiteral(distance)


def as_angle(angle: float | Angle) -> Angle:
    if isinstance(angle, Angle):
        return angle
    return AngleLiteral(angle)


def as_point(point: tuple[float, float] | Point) -> Point:
    if isinstance(point, Point):
        return point
    x, y = point
    return CartesianPoint(x, y)


def as_polar_point(point: tuple[float | Angle, float | Distance] | Point) -> Point:
    if isinstance(point, Point):
        return point
    angle, radius = point
    return PolarPoint(as_angle(angle), as_distance(radius))


class PointCreator:
    def __init__(self, current_point: Point, done):
        self.current_point = current_point
        self._done = done

    def _return_point(self, point: Point):
        return self._done(point)

    def go_to(self, x, y):
        p = CartesianPoint(x, y)
        return self._return_point(p)

    def go_to_polar(self, angle, radius):
        p = PolarPoint(as_angle(angle), as_distance(radius))
        return self._return_point(p)

    def go_to_point(self, point):
        return self._return_point(point)

    def move_by(self, x, y):
        p = self.current_point + CartesianPoint(x, y).vec()
        return self._return_point(p)

    def move_by_polar(self, angle, radius):
        p = self.current_point + PolarPoint(as_angle(angle), as_distance(radius)).vec()
        return self._return_point(p)

    def horizontal_move_by(self, x):
        return self.move_by(x, 0)

    def horizontal_go_to(self, x):
        return self.go_to(x, 0)

    def vertical_move_by(self, y):
        return self.move_by(0, y)

    def vertical_go_to(self, y):
        return self.go_to(0, y)

    def close(self) -> Shape:
        return self._done(None)


class EdgeMaker:
    def __init__(self, current_point: Point, done):
        self._done = done
        self.current_point = current_point

    def line(self):
        return self._done(Line())

    def arc(self, bulge):
        return self._done(ArcBulge(bulge))

    def arc_tangent_start(self, angle):
        return self._done(ArcTangentStart(as_angle(angle)))

    def arc_tangent_end(self, angle):
        return self._done(ArcTangentEnd(as_angle(angle)))

    def arc_smooth_start(self):
        return self._done(ArcWithSmoothStart())

    def arc_smooth_end(self):
        return self._done(ArcWithSmoothEnd())

    def biarc(self, start_angle, end_angle):
        return self._done(Biarc(as_angle(start_angle), as_angle(end_angle), 0.5))

    def biarc_with_param(self, start_angle, end_angle, p=0.5):
        return self._done(Biarc(as_angle(start_angle), as_angle(end_angle), p))

    def biarc_smooth_start(self, end_angle):
        return self._done(BiarcWithSmoothStart(as_angle(end_angle), 0.5))

    def biarc_smooth_end(self, start_angle):
        return self._done(BiarcWithSmoothEnd(as_angle(start_angle), 0.5))

    def biarc_smooth_extremities(self):
        return self._done(BiarcWithSmoothExtremities(0.5))


def draw(origin: tuple[float, float] = (0, 0)):
    current_point = as_point(origin)
    path: Path = PathStart(current_point)

    def line_done(line):
        def point_done(point):
            nonlocal path
            if point is None:
                return sketch_shape(PathClose(path, line))

            nonlocal current_point
            current_point = point
            path = PathEdge(path, line, point)
            return EdgeMaker(current_point, line_done)

        nonlocal current_point
        return PointCreator(current_point, point_done)

    return EdgeMaker(current_point, line_done)


class LossMaker:
    def __init__(self):
        self.constraints = []

    def fit_distance(self, distance, target, tol=1e-3):
        self.constraints.append(
            ConstraintOnDistance(as_distance(distance), target, tol)
        )
        return self

    def fit_angle(self, angle, target, tol=1e-3):
        self.constraints.append(ConstraintOnAngle(as_angle(angle), target, tol))
        return self

    def fit_point(self, point, target, tol=1e-3):
        self.constraints.append(
            ConstraintOnPointCoincidence(as_point(point), as_point(target), tol)
        )
        return self

    def create_unit(self):
        loss = sum([constraint_loss(c) for c in self.constraints], ZERO)
        return Unit().registerLoss(loss)


def loss():
    return LossMaker()
