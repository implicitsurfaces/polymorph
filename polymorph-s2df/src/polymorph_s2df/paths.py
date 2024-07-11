from typing import Sequence

from polymorph_num import ops
from polymorph_num.expr import PI, TAU, Expr, Num, as_expr
from polymorph_num.vec import ValVec, Vec2, as_vec2

from .operations import Shape
from .utils import min_iterable, repr_point, sum_iterable


class PathSegment(Shape):
    @property
    def first_point(self) -> Vec2:
        raise NotImplementedError()

    @property
    def last_point(self) -> Vec2:
        raise NotImplementedError()

    def astuple(self):
        raise NotImplementedError()

    def __hash__(self):
        return hash(self.astuple())

    def distance(self, x: Num, y: Num) -> Expr:
        raise NotImplementedError()

    def winding_number(self, p: Vec2) -> Expr:
        raise NotImplementedError()


class LineSegment(PathSegment):
    def __init__(self, start: ValVec, end: ValVec):
        super().__init__()
        self.start = as_vec2(start)
        self.end = as_vec2(end)

    def astuple(self):
        return (self.start, self.end)

    def __eq__(self, other):
        return isinstance(other, LineSegment) and self.astuple() == other.astuple()

    def __hash__(self):
        return hash(self.astuple())

    @property
    def first_point(self) -> Vec2:
        return self.start

    @property
    def last_point(self) -> Vec2:
        return self.end

    def __repr__(self):
        return f"LineSegment({repr_point(self.start)}, {repr_point(self.end)})"

    @property
    def segment(self):
        return self.end - self.start

    def winding_number(self, p: Vec2) -> Expr:
        a = self.start - p
        b = self.end - p

        return ops.atan2(a.cross(b), a.dot(b))

    def distance(self, x: Num, y: Num) -> Expr:
        p = Vec2(x, y)
        start_to_p = p - self.start

        parametric_position = start_to_p.dot(self.segment) / self.segment.dot(
            self.segment
        )
        clamped_position = ops.clamp(parametric_position, 0, 1)

        projected_point = self.start + self.segment.scale(clamped_position)
        return (p - projected_point).norm()


def normalize_angle(q):
    return ((q % TAU) + TAU) % TAU


def winding_number_indefinite_integral(t, radius, x, y):
    R2 = radius * radius
    half_t = t / 2
    sin_t = half_t.sin()
    cos_t = half_t.cos()

    term1 = R2 * sin_t + 2 * radius * (x * sin_t - y * cos_t)
    term2 = (x * x + y * y) * sin_t

    numerator = (term1 + term2) / cos_t
    denominator = R2 - x * x - y * y

    arctan_term = ops.atan2(denominator, numerator)

    return radius * (arctan_term / radius + t / (radius * 2))


def angular_distance(start_angle, end_angle, orientation_sign):
    raw_distance = orientation_sign * (end_angle - start_angle)
    return (raw_distance + TAU) % TAU


class ArcSegment(PathSegment):
    start_angle: Expr
    end_angle: Expr
    radius: Expr

    def __init__(self, start_angle, end_angle, radius, orientation_sign):
        super().__init__()
        self.start_angle = normalize_angle(as_expr(start_angle))
        self.end_angle = normalize_angle(as_expr(end_angle))
        self.radius = as_expr(radius)
        self.orientation_sign = as_expr(orientation_sign)

    def astuple(self):
        return (self.start_angle, self.end_angle, self.radius, self.orientation_sign)

    def __eq__(self, other):
        return isinstance(other, ArcSegment) and self.astuple() == other.astuple()

    def __hash__(self):
        return hash(self.astuple())

    @property
    def first_point(self) -> Vec2:
        return Vec2(
            self.radius * self.start_angle.cos(), self.radius * self.start_angle.sin()
        )

    @property
    def last_point(self) -> Vec2:
        return Vec2(
            self.radius * self.end_angle.cos(), self.radius * self.end_angle.sin()
        )

    def winding_number(self, p: Vec2) -> Expr:
        end_angle_integral = winding_number_indefinite_integral(
            self.end_angle, self.radius, p.x, p.y
        )
        start_angle_integral = winding_number_indefinite_integral(
            self.start_angle, self.radius, p.x, p.y
        )

        pi_integral = winding_number_indefinite_integral(PI, self.radius, p.x, p.y)
        min_pi_integral = winding_number_indefinite_integral(-PI, self.radius, p.x, p.y)

        # First, we consider the case where the angles are oriented counter
        # clockwise
        #
        # We need to consider three cases: - both angles are smaller than pi
        # - both angles are greater than pi - one angle is smaller than pi and
        # the other is greater than pi
        #
        # If the angles are on the same side of pi, we cross the pi line if end
        # angle is small than the start angle. This is the same for when they
        # are both bigger and small than pi - so we really only have two cases,
        # either the angles are on the same side of pi or not.
        #
        # We can use the sign of the product of the differences to determine if
        # the angles are on the same side of pi.
        #
        # Then, as the cases are the inverse of each other, we can use the sign
        # of the product of the differences
        #
        # An angle is smaller than pi if the difference between the angle and
        # pi is positive and greater than pi if the difference is negative
        #
        # When the angles are on different sides of pi, we cross pi if the end
        # angles is bigger than the start angle.
        # As this is the opposite of the case where the angles are on the same
        # side of pi, we can use the sign of the product of the differences to
        # determine if the angles are on the same side of pi.

        is_crossing_pi = (
            (PI - self.start_angle)
            * (PI - self.end_angle)
            * (self.end_angle - self.start_angle)
        )

        # We then need to take into account the case where the angles are
        # clockwise. In that case we need to make two changes:
        #
        # - the angles are inverted (i.e. the is_crossing_pi needs to invert
        # its sign)
        #
        # - the integral needs to be inverted (i.e. we need to subtract the
        # integral from the start to the end)

        pi_crossing_correction = (
            ops.if_lt(
                is_crossing_pi * self.orientation_sign,
                0,
                min_pi_integral - pi_integral,
                0,
            )
            * self.orientation_sign
        )

        return -((end_angle_integral - start_angle_integral) + pi_crossing_correction)

    def distance(self, x: Num, y: Num) -> Expr:
        p = Vec2(x, y)
        angular_length = angular_distance(
            self.start_angle, self.end_angle, self.orientation_sign
        )

        angle_position = normalize_angle(ops.atan2(y, x))

        parametric_position = (
            angular_distance(self.start_angle, angle_position, self.orientation_sign)
            / angular_length
        )

        clamped_position = self.orientation_sign * ops.clamp(parametric_position, 0, 1)
        clamped_angle = normalize_angle(
            self.start_angle + clamped_position * angular_length
        )
        projected_point = Vec2(
            self.radius * clamped_angle.cos(), self.radius * clamped_angle.sin()
        )

        start_point = Vec2(
            self.radius * self.start_angle.cos(), self.radius * self.start_angle.sin()
        )
        end_point = Vec2(
            self.radius * self.end_angle.cos(), self.radius * self.end_angle.sin()
        )

        return min_iterable(
            (p - point).norm() for point in (projected_point, start_point, end_point)
        )


class TranslatedSegment(PathSegment):
    def __init__(self, segment: PathSegment, translation: ValVec):
        super().__init__()
        self.segment = segment
        self.translation = as_vec2(translation)

    def astuple(self):
        return (self.segment, self.translation)

    def __eq__(self, other):
        return (
            isinstance(other, TranslatedSegment) and self.astuple() == other.astuple()
        )

    def __hash__(self):
        return hash(self.astuple())

    def __repr__(self):
        return f"TranslatedSegment({self.segment}, {repr_point(self.translation)})"

    @property
    def first_point(self):
        return self.segment.first_point + self.translation

    @property
    def last_point(self):
        return self.segment.last_point + self.translation

    def distance(self, x, y):
        return self.segment.distance(x - self.translation.x, y - self.translation.y)

    def winding_number(self, p):
        return self.segment.winding_number(p - self.translation)


class InversedSegment(PathSegment):
    def __init__(self, segment: PathSegment):
        super().__init__()
        self.segment = segment

    def astuple(self):
        return (self.segment,)

    def __eq__(self, other):
        return isinstance(other, InversedSegment) and self.astuple() == other.astuple()

    def __hash__(self):
        return hash(self.astuple())

    def __repr__(self):
        return f"InversedSegment({self.segment})"

    @property
    def first_point(self) -> Vec2:
        return self.segment.first_point

    @property
    def last_point(self) -> Vec2:
        return self.segment.last_point

    def winding_number(self, p: Vec2):
        return -self.segment.winding_number(p)


class ClosedPath(Shape):
    def __init__(self, segments: Sequence[PathSegment]):
        super().__init__()
        self.segments = segments

    def astuple(self):
        return tuple(self.segments)

    def __hash__(self):
        return hash(self.astuple())

    def __eq__(self, other):
        return isinstance(other, ClosedPath) and self.astuple() == other.astuple()

    def _min_distance_to_points(self, p):
        return min_iterable(
            (p - segment.first_point).norm() for segment in self.segments
        )

    def winding_number(self, p: Vec2) -> Expr:
        return (
            sum_iterable(segment.winding_number(p) for segment in self.segments) / TAU
        )

    def distance(self, x: Num, y: Num) -> Expr:
        minimum_distance = min_iterable(
            segment.distance(x, y) for segment in self.segments
        )

        # We need to map the winding number such that outside the path it is 1
        # and inside it is -1
        p = Vec2(x, y)
        current_sign = 1 - ops.min(self.winding_number(p).abs(), 1) * 2

        return minimum_distance * current_sign
