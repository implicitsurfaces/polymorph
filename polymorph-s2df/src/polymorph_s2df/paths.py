from functools import cached_property
from typing import Sequence

from polymorph_num import ops
from polymorph_num.expr import PI, TAU, Expr, Num, as_expr
from polymorph_num.vec import ValVec, Vec2, as_vec2

from polymorph_s2df.bounding_box import BoundingBox, bounding_box_from_points

from .operations import Shape
from .utils import (
    angular_distance,
    diamond_atan,
    max_iterable,
    min_iterable,
    normalize_angle,
    repr_point,
    sum_iterable,
)


class PathSegment(Shape):
    def astuple(self):
        raise NotImplementedError()

    def __hash__(self):
        return hash(self.astuple())

    def end_tangent(self) -> Expr:
        raise NotImplementedError()

    def distance(self, x: Num, y: Num) -> Expr:
        raise NotImplementedError()

    def winding_number(self, p: Vec2) -> Expr:
        raise NotImplementedError()

    def bounding_box(self):
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

    def __repr__(self):
        return f"LineSegment({repr_point(self.start)}, {repr_point(self.end)})"

    @cached_property
    def segment(self):
        return self.end - self.start

    def end_tangent(self):
        return self.segment.atan2()

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

    def bounding_box(self):
        return BoundingBox(
            ops.min(self.start.x, self.end.x),
            ops.min(self.start.y, self.end.y),
            ops.max(self.start.x, self.end.x),
            ops.max(self.start.y, self.end.y),
        )


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

    @cached_property
    def first_point(self) -> Vec2:
        return Vec2(
            self.radius * self.start_angle.cos(), self.radius * self.start_angle.sin()
        )

    def end_tangent(self) -> Expr:
        return self.end_angle

    @cached_property
    def last_point(self) -> Vec2:
        return Vec2(
            self.radius * self.end_angle.cos(), self.radius * self.end_angle.sin()
        )

    @cached_property
    def angular_length(self):
        return angular_distance(self.start_angle, self.end_angle, self.orientation_sign)

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
        angle_position = normalize_angle(ops.atan2(y, x))

        parametric_position = (
            angular_distance(self.start_angle, angle_position, self.orientation_sign)
            / self.angular_length
        )

        clamped_position = self.orientation_sign * ops.clamp(parametric_position, 0, 1)
        clamped_angle = normalize_angle(
            self.start_angle + clamped_position * self.angular_length
        )
        projected_point = Vec2(
            self.radius * clamped_angle.cos(), self.radius * clamped_angle.sin()
        )

        return min_iterable(
            (p - point).norm()
            for point in (projected_point, self.first_point, self.last_point)
        )

    def bounding_box(self):
        return BoundingBox(
            -self.radius,
            -self.radius,
            self.radius,
            self.radius,
        )


class BulgingSegment(PathSegment):
    start: Vec2
    end: Vec2
    bulge: Expr

    def __init__(self, start: ValVec, end: ValVec, bulge: Num):
        super().__init__()
        self.start = as_vec2(start)
        self.end = as_vec2(end)
        self.bulge = as_expr(bulge)

    def astuple(self):
        return (self.start, self.end, self.bulge)

    def __eq__(self, other):
        return isinstance(other, BulgingSegment) and self.astuple() == other.astuple()

    def __hash__(self):
        return hash(self.astuple())

    @cached_property
    def chord(self):
        return self.end - self.start

    @cached_property
    def center(self):
        chord_center = (self.start + self.end) / 2
        bb = (self.bulge - (1 / self.bulge)) / 4
        return chord_center - self.chord.perp().scale(bb)

    @cached_property
    def radius(self):
        return self.chord.norm() / 4 * (self.bulge + (1 / self.bulge)).abs()

    @cached_property
    def s_value(self):
        return 2 * self.bulge / (1 + self.bulge**2)

    @cached_property
    def c_value(self):
        return (1 - self.bulge**2) / (1 + self.bulge**2)

    @cached_property
    def start_diangle(self):
        return diamond_atan(self.start.x - self.center.x, self.start.y - self.center.y)

    @cached_property
    def end_diangle(self):
        return diamond_atan(self.end.x - self.center.x, self.end.y - self.center.y)

    def distance(self, x: Num, y: Num) -> Expr:
        p = Vec2(x, y)

        in_sector_distance = ((p - self.center).norm() - self.radius).abs()
        out_sector_distance = ops.min((p - self.start).norm(), (p - self.end).norm())

        center_to_p = p - self.center

        sign = self.bulge.sign()

        p_diangle = sign * diamond_atan(center_to_p.x, center_to_p.y)
        s_diangle = sign * self.start_diangle
        e_diangle = sign * self.end_diangle

        min_diangle = min_iterable((s_diangle, e_diangle, p_diangle))
        max_diangle = max_iterable((s_diangle, e_diangle, p_diangle))

        def if_not_extrema(value, if_extrema, if_not_extrema):
            return ops.if_eq(
                value,
                min_diangle,
                if_extrema,
                ops.if_eq(value, max_diangle, if_extrema, if_not_extrema),
            )

        return ops.if_lt(
            s_diangle,
            p_diangle,
            if_not_extrema(e_diangle, in_sector_distance, out_sector_distance),
            if_not_extrema(e_diangle, out_sector_distance, in_sector_distance),
        )


class TranslatedSegment(PathSegment):
    def __init__(self, segment: PathSegment, translation: ValVec):
        super().__init__()
        self.segment = segment
        self.translation = as_vec2(translation)

    def astuple(self):
        return (self.segment, self.translation)

    def end_tangent(self) -> Expr:
        return self.segment.end_tangent()

    def __eq__(self, other):
        return (
            isinstance(other, TranslatedSegment) and self.astuple() == other.astuple()
        )

    def __hash__(self):
        return hash(self.astuple())

    def __repr__(self):
        return f"TranslatedSegment({self.segment}, {repr_point(self.translation)})"

    def distance(self, x, y):
        return self.segment.distance(x - self.translation.x, y - self.translation.y)

    def winding_number(self, p):
        return self.segment.winding_number(p - self.translation)

    def bounding_box(self):
        return self.segment.bounding_box().translate(
            self.translation.x, self.translation.y
        )


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

    def bounding_box(self):
        return bounding_box_from_points(
            [
                point
                for segment in self.segments
                for point in (segment.bounding_box().min, segment.bounding_box().max)
            ]
        )
