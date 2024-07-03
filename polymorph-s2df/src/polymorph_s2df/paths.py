from typing import Sequence

from polymorph_num import ops
from polymorph_num.expr import TAU, Expr, as_expr
from polymorph_num.vec import Vec2, as_vec2

from .operations import Shape
from .utils import min_iterable, repr_point, smooth_clamp_mask, sum_iterable


class PathSegment:
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

    def distance_and_mask(self, p: Vec2) -> tuple[Expr, Expr]:
        raise NotImplementedError()

    def winding_number(self, p: Vec2) -> Expr:
        raise NotImplementedError()


class LineSegment(PathSegment):
    def __init__(self, start: Vec2, end: Vec2):
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

    def distance_and_mask(self, p: Vec2) -> tuple[Expr, Expr]:
        start_to_p = p - self.start

        parametric_position = start_to_p.dot(self.segment) / self.segment.dot(
            self.segment
        )

        mask = smooth_clamp_mask(parametric_position, 0, 1)

        projected_point = self.start + self.segment.scale(parametric_position)
        return (
            (p - projected_point).norm() * mask,
            mask,
        )


def normalize_angle(q):
    return (q + TAU) % TAU


def winding_number_indefinite_integral(t, radius, x, y):
    half_t = t / 2
    sin_t = half_t.sin()
    cos_t = half_t.cos()

    numerator = y * cos_t + (-radius - x) * sin_t
    denominator = (x - radius) * cos_t + y * sin_t

    arctan_term = ops.atan2(denominator, numerator)

    return half_t + arctan_term


def winding_number_indefinite_integral_2(t, radius, x, y):
    R2 = radius * radius
    half_t = t / 2
    sin_t = half_t.sin()
    cos_t = half_t.cos()

    term1 = R2 * sin_t + as_expr(2) * radius * (x * sin_t - y * cos_t)
    term2 = (x * x + y * y) * sin_t

    numerator = (term1 + term2) / cos_t
    denominator = R2 - x * x - y * y

    arctan_term = ops.atan2(denominator, numerator)

    return radius * (arctan_term / radius + t / (radius * 2))


class ArcSegment(PathSegment):
    start_angle: Expr
    end_angle: Expr
    radius: Expr

    def __init__(self, start_angle, end_angle, radius):
        super().__init__()
        self.start_angle = normalize_angle(as_expr(start_angle))
        self.end_angle = normalize_angle(as_expr(end_angle))
        self.radius = as_expr(radius)

    def astuple(self):
        return (self.start_angle, self.end_angle, self.radius)

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
        return winding_number_indefinite_integral(
            (self.end_angle), self.radius, p.x, p.y
        ) - winding_number_indefinite_integral(
            (self.start_angle), self.radius, p.x, p.y
        )

    def distance_and_mask(self, p: Vec2) -> tuple[Expr, Expr]:
        angular_length = (self.end_angle) - self.start_angle

        angle_position = normalize_angle(ops.atan2(p.y, p.x))

        parametric_position = (angle_position - self.start_angle) / angular_length
        mask = smooth_clamp_mask(parametric_position, 0, 1)

        parametric_distance = (p.norm() - self.radius) * mask

        return parametric_distance.abs(), mask


class TranslatedSegment(PathSegment):
    def __init__(self, segment: PathSegment, translation: Vec2):
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

    def tree_flatten(self):
        return (self.segment, self.translation), None

    @property
    def first_point(self):
        return self.segment.first_point + self.translation

    @property
    def last_point(self):
        return self.segment.last_point + self.translation

    def distance_and_mask(self, p):
        return self.segment.distance_and_mask(p - self.translation)

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

    def distance_and_mask(self, p: Vec2):
        return self.segment.distance_and_mask(p)

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

    def distance(self, x: Expr, y: Expr) -> Expr:
        # The gist of the algorithm is to combine the distance to each segment
        # and to each point between the segments. Segments cannot apply to the
        # whole space, so we need to combine them with "masks".

        # We use a mask instead of an if condition as jax jit does not support it

        p = Vec2(x, y)

        distances_and_masks = [
            segment.distance_and_mask(p) for segment in self.segments
        ]

        current_distance, current_mask = distances_and_masks[0]
        minimum_distance = current_distance

        # We combine the distance for all the segments
        for distance, mask in distances_and_masks[1:]:
            distance = distance

            # first we handle the case where the distance can be defined by
            # its distance to both segment
            current_distance = ops.min(minimum_distance * mask, distance * current_mask)

            # if only one of the segment can apply we use it
            current_distance += minimum_distance * (as_expr(1) - mask) + distance * (
                as_expr(1) - current_mask
            )

            current_mask = as_expr(1) - (
                (as_expr(1) - mask) * (as_expr(1) - current_mask)
            )
            minimum_distance = current_distance

        points_distance = self._min_distance_to_points(p)
        minimum_distance = (
            ops.min(minimum_distance, points_distance)
            + (as_expr(1) - current_mask) * points_distance
        )

        total_winding_number = (
            sum_iterable(segment.winding_number(p) for segment in self.segments) / TAU
        )
        # We need to map the winding number such that outside the path it is 1
        # and inside it is -1

        current_sign = as_expr(1) - ops.min(total_winding_number.abs(), as_expr(1)) * 2

        return minimum_distance * current_sign
