from dataclasses import dataclass
from typing import Iterable, Sequence

from polymorph_num import ops
from polymorph_num.expr import Expr, as_expr, Infinity
from polymorph_num.vec import Vec2


def norm(x: Expr, y: Expr) -> Expr:
    return (x * x + y * y).sqrt()


def softminus(x: Expr) -> Expr:
    return x - x.softplus()


def smooth_clamp_mask(x, low=0.0, high=1.0, softness=1e-6):
    """A smooth implementation of a mask between the boundary values"""
    lower_transition = as_expr(0.5) * (as_expr(1) + ((x - low) / softness).tanh())
    upper_transition = as_expr(0.5) * (as_expr(1) - ((x - high) / softness).tanh())
    return lower_transition * upper_transition


def sum_iterable(values: Iterable[Expr]) -> Expr:
    x = as_expr(0)
    for item in values:
        x += item
    return x


def min_iterable(values: Iterable[Expr]) -> Expr:
    x = Infinity
    for item in values:
        x = ops.min(x, item)
    return x


@dataclass(frozen=True)
class Shape:
    def is_inside(self, x: Expr, y: Expr, scale=100) -> Expr:
        return as_expr(1) - (self.distance(x, y) * scale).sigmoid()

    def distance(self, x: Expr, y: Expr) -> Expr:
        raise NotImplementedError()


@dataclass(frozen=True)
class Translation(Shape):
    offset: Vec2
    shape: Shape

    def distance(self, x: Expr, y: Expr) -> Expr:
        return self.shape.distance(x - self.offset.x, y - self.offset.y)


@dataclass(frozen=True)
class Circle(Shape):
    radius: Expr

    def distance(self, x: Expr, y: Expr) -> Expr:
        return norm(x, y) - self.radius


@dataclass(frozen=True)
class Union(Shape):
    a: Shape
    b: Shape

    def distance(self, x: Expr, y: Expr) -> Expr:
        return ops.min(self.a.distance(x, y), self.b.distance(x, y))


@dataclass(frozen=True)
class Box(Shape):
    width: Expr
    height: Expr

    def distance(self, x: Expr, y: Expr) -> Expr:
        q_x = x.smoothabs() - self.width / 2
        q_y = y.smoothabs() - self.height / 2

        return norm(q_x.softplus(), q_y.softplus()) + softminus(ops.max(q_x, q_y))


@dataclass(frozen=True)
class PathSegment:
    start: Vec2
    end: Vec2

    def distance_and_mask(self, p: Expr) -> Expr:
        raise NotImplementedError()

    def winding_number(self, p: Vec2) -> Expr:
        raise NotImplementedError()


@dataclass(frozen=True)
class LineSegment(PathSegment):
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


@dataclass(frozen=True)
class ClosedPath(Shape):
    segments: Sequence[PathSegment]

    def _min_distance_to_points(self, p):
        return min_iterable((p - segment.start).norm() for segment in self.segments)

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

        total_winding_number = sum_iterable(
            segment.winding_number(p) for segment in self.segments
        )
        # We need to map the winding number such that outside the path it is 1
        # and inside it is -1

        current_sign = as_expr(1) - ops.min(total_winding_number.abs(), as_expr(1)) * 2

        return minimum_distance * current_sign
