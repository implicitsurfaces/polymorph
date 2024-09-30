from .nodes import (
    Angle,
    AngleLiteral,
    ArcBulge,
    ArcTangentEnd,
    ArcTangentStart,
    CartesianPoint,
    Distance,
    DistanceLiteral,
    Line,
    Path,
    PathClose,
    PathEdge,
    PathStart,
    Point,
    PolarPoint,
)


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
        p = self.current_point + CartesianPoint(x, y)
        return self._return_point(p)

    def move_by_polar(self, angle, radius):
        p = self.current_point + PolarPoint(as_angle(angle), as_distance(radius))
        return self._return_point(p)

    def horizontal_move_by(self, x):
        return self.move_by(x, 0)

    def horizontal_go_to(self, x):
        return self.go_to(x, 0)

    def vertical_move_by(self, y):
        return self.move_by(0, y)

    def vertical_go_to(self, y):
        return self.go_to(0, y)

    def close(self):
        return self._done(None)


class LineCreator:
    def __init__(self, done):
        self._done = done

    def line(self):
        return self._done(Line())

    def arc(self, bulge):
        return self._done(ArcBulge(bulge))

    def arc_tangent_start(self, angle):
        return self._done(ArcTangentStart(as_angle(angle)))

    def arc_tangent_end(self, angle):
        return self._done(ArcTangentEnd(as_angle(angle)))


def draw(origin: tuple[float, float] = (0, 0)):
    current_point = as_point(origin)
    path: Path = PathStart(current_point)

    def line_done(line):
        def point_done(point):
            nonlocal path
            if point is None:
                return PathClose(path, line)

            nonlocal current_point
            current_point = point
            path = PathEdge(path, point, line)
            return LineCreator(line_done)

        nonlocal current_point
        return PointCreator(current_point, point_done)

    return LineCreator(line_done)
