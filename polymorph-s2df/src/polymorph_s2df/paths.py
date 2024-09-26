from functools import cached_property
from typing import Sequence

from polymorph_num import ops
from polymorph_num.angle import (
    Angle,
    polar_angle,
    polar_angle_from_vec,
    two_vectors_angle,
)
from polymorph_num.expr import TAU, ZERO, Expr, Num, as_expr
from polymorph_num.vec import ValVec, Vec2, as_vec2

from polymorph_s2df.bounding_box import BoundingBox, bounding_box_from_points

from .operations import Shape
from .utils import (
    max_iterable,
    min_iterable,
    repr_point,
)


class SolidAngle:
    _val: Expr

    def __init__(self, value: Expr):
        self._val = value

    def __add__(self, other: "SolidAngle"):
        return SolidAngle(self._val + other._val)

    def __neg__(self):
        return SolidAngle(-self._val)

    def __sub__(self, other: "SolidAngle"):
        return SolidAngle(self._val - other._val)

    def half(self):
        return SolidAngle(self._val / 2)

    def flip_sign(self, sign):
        return SolidAngle(self._val * sign)

    def as_rad(self):
        return self._val * TAU

    def full_turns(self):
        return self._val


def is_negative(x: Expr):
    return ops.if_lt(x, 0, 1, 0)


def is_positive(x: Expr):
    return ops.if_gt(x, 0, 1, 0)


def as_solid_angle(angle: Angle):
    return SolidAngle(angle.as_rad() / TAU)


class PathSegment(Shape):
    def astuple(self):
        raise NotImplementedError()

    def __hash__(self):
        return hash(self.astuple())

    def end_tangent(self) -> Angle:
        raise NotImplementedError()

    def distance(self, x: Num, y: Num) -> Expr:
        raise NotImplementedError()

    def winding_number(self, p: Vec2) -> Expr:
        return self.solid_angle(p).full_turns()

    def solid_angle(self, p: Vec2) -> SolidAngle:
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
        return polar_angle(self.segment.x, self.segment.y)

    def solid_angle(self, p: Vec2) -> SolidAngle:
        a = self.start - p
        b = self.end - p

        return as_solid_angle(two_vectors_angle(a, b))

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


def negative_sign(x: Expr):
    return ops.if_lt(x, 0, -1, 1)


def winding_number_indefinite_integral(t: Angle, radius: Expr, x: Expr, y: Expr):
    R2 = radius * radius
    half_t = t.half()
    sin_t = half_t.sin()
    cos_t = half_t.cos()

    term1 = 2 * radius * y
    x_r = x + radius
    term2 = (x_r * x_r + y * y) * sin_t / cos_t

    angle_x = term2 - term1
    angle_y = R2 - x * x - y * y

    return as_solid_angle(polar_angle(angle_x, angle_y)) + as_solid_angle(t).half()


def winding_number_at_pi(radius: Expr, x: Expr, y: Expr):
    dist_to_center = radius * radius - (x * x + y * y)
    return dist_to_center.sign() / 2


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
    def start_sort_angle(self):
        return polar_angle(
            self.start.x - self.center.x, self.start.y - self.center.y
        ).as_sort_value()

    @cached_property
    def end_sort_angle(self):
        return polar_angle(
            self.end.x - self.center.x, self.end.y - self.center.y
        ).as_sort_value()

    def start_tangent(self) -> Angle:
        return polar_angle_from_vec(
            self.chord.scale(self.c_value) - self.chord.perp().scale(self.s_value)
        )

    def end_tangent(self) -> Angle:
        return polar_angle_from_vec(
            self.chord.scale(self.c_value) + self.chord.perp().scale(self.s_value)
        )

    def distance(self, x: Num, y: Num) -> Expr:
        p = Vec2(x, y)

        in_sector_distance = ((p - self.center).norm() - self.radius).abs()
        out_sector_distance = ops.min((p - self.start).norm(), (p - self.end).norm())

        orientation = self.bulge.sign()

        # We want to decide if we return the in_sector_distance or the out_sector_distance
        # We can do this by comparing the angles of the start, end and the point
        #
        # The gist of it is that if the sort order of the points is a even permutation of
        # start-point-end (i.e. the point is in the sector), we return the in_sector_distance
        # otherwise we return the out_sector_distance

        p_sort_val = orientation * polar_angle_from_vec(p - self.center).as_sort_value()
        s_sort_val = orientation * self.start_sort_angle
        e_sort_val = orientation * self.end_sort_angle

        min_sort_val = min_iterable((s_sort_val, e_sort_val, p_sort_val))
        max_sort_val = max_iterable((s_sort_val, e_sort_val, p_sort_val))

        def if_not_extrema(value, if_extrema, if_not_extrema):
            return ops.if_eq(
                value,
                min_sort_val,
                if_extrema,
                ops.if_eq(value, max_sort_val, if_extrema, if_not_extrema),
            )

        return ops.if_lt(
            s_sort_val,
            p_sort_val,
            if_not_extrema(e_sort_val, in_sector_distance, out_sector_distance),
            if_not_extrema(e_sort_val, out_sector_distance, in_sector_distance),
        )

    def solid_angle(self, p: Vec2) -> SolidAngle:
        start_angle = polar_angle_from_vec(self.start - self.center)
        end_angle = polar_angle_from_vec(self.end - self.center)
        p_vec = p - self.center

        end_angle_integral = winding_number_indefinite_integral(
            end_angle, self.radius, p_vec.x, p_vec.y
        )
        start_angle_integral = winding_number_indefinite_integral(
            start_angle, self.radius, p_vec.x, p_vec.y
        )

        # First, we consider the case where the angles are oriented counter
        # clockwise
        #
        # We need to consider three cases: - both angles are smaller than pi
        # - both angles are greater than pi - one angle is smaller than pi and
        # the other is greater than pi
        #
        # If the angles are on the same side of pi, we cross the pi line if the
        # end angle is smaller than the start angle. This is the same for when
        # they are both bigger and small than pi - so we really only have two
        # cases, either the angles are on the same side of pi or not.
        #
        # We can use the sign of the product of the sin of the angles to check
        # if the angles are on the same side of pi.

        same_side_of_pi_sign = start_angle.sin() * end_angle.sin()

        # Then, as the cases are the inverse of each other, we can use the sign
        # of the product of the differences
        #
        # An angle is smaller than pi if the difference between the angle and
        # pi is positive and greater than pi if the difference is negative
        #
        # When the angles are on different sides of pi, we cross pi if the end
        # angles is bigger than the start angle. As this is the opposite of the
        # case where the angles are on the same side of pi, we can use the sign
        # of the product of the differences to determine if the angles are on
        # the same side of pi.

        span_sign = end_angle.as_sort_value() - start_angle.as_sort_value()

        # We then need to take into account the case where the angles are
        # clockwise. In that case we need to make two changes:
        #
        # - the angles are inverted (i.e. the is_crossing_pi needs to invert
        # its sign)
        #
        # - the integral needs to be inverted (i.e. we need to subtract the
        # integral from the start to the end)

        orientation_sign = self.bulge.sign()

        is_crossing_pi = same_side_of_pi_sign * span_sign * orientation_sign

        pi_crossing_correction_turns = ops.if_lt(
            is_crossing_pi,
            0,
            orientation_sign * winding_number_at_pi(self.radius, p_vec.x, p_vec.y),
            0,
        )

        # Note that the correction is either 0, π or -, so we create a solid angle
        # with a known number of turns
        correction_solid_angle = SolidAngle(pi_crossing_correction_turns)

        # This seems reversed, but it is correct. The sign convention needs this
        return start_angle_integral - end_angle_integral + correction_solid_angle

    def bounding_box(self):
        return BoundingBox(
            self.center.x - self.radius,
            self.center.y - self.radius,
            self.center.x + self.radius,
            self.center.y + self.radius,
        )


class TranslatedSegment(PathSegment):
    def __init__(self, segment: PathSegment, translation: ValVec):
        super().__init__()
        self.segment = segment
        self.translation = as_vec2(translation)

    def astuple(self):
        return (self.segment, self.translation)

    def end_tangent(self) -> Angle:
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

    def solid_angle(self, p: Vec2) -> SolidAngle:
        return self.segment.solid_angle(p - self.translation)

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

    def solid_angle(self, p: Vec2) -> SolidAngle:
        return sum(
            (segment.solid_angle(p) for segment in self.segments), SolidAngle(ZERO)
        )

    def winding_number(self, p: Vec2) -> Expr:
        return self.solid_angle(p).full_turns()

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
