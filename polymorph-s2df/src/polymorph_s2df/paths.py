import jax.numpy as jnp
import jax.lax

from .operations import Shape
from .utils import clamp_mask, indent_shape


class LineSegment:
    def __init__(self, start, end):
        self.start = start
        self.end = end

        self.segment = end - start
        self.segment_length_square = jnp.dot(self.segment, self.segment)

        length = jnp.sqrt(self.segment_length_square)
        self.start_tangent = self.segment / length
        self.end_tangent = -self.segment / length

    def __repr__(self):
        return f"LineSegment({self.start}, {self.end})"

    def distance_and_mask(self, p):
        start_to_p = p - self.start

        parametric_position = (
            jnp.dot(start_to_p, self.segment) / self.segment_length_square
        )

        mask = clamp_mask(parametric_position, 0, 1)

        projected_point = self.start + self.segment * parametric_position
        return jnp.sign(jnp.cross(start_to_p, self.segment)) * jnp.linalg.norm(
            p - projected_point
        ) * mask, mask


def normalize_angle(q):
    return jnp.mod(q + 2 * jnp.pi, 2 * jnp.pi)


class ArcSegment:
    def __init__(self, start_angle, end_angle, radius):
        self.start_angle = normalize_angle(start_angle)
        self.end_angle = normalize_angle(end_angle)
        self.radius = radius

        self.start = jnp.array(
            [radius * jnp.cos(start_angle), radius * jnp.sin(start_angle)]
        )
        self.end = jnp.array([radius * jnp.cos(end_angle), radius * jnp.sin(end_angle)])

        self.angular_length = end_angle - start_angle

        self.start_tangent = jnp.sign(self.angular_length) * jnp.array(
            [-jnp.sin(start_angle), jnp.cos(start_angle)]
        )
        self.end_tangent = jnp.sign(self.angular_length) * jnp.array(
            [jnp.sin(end_angle), -jnp.cos(end_angle)]
        )

    def __repr__(self):
        return f"ArcSegment({self.start_angle}, {self.end_angle}, {self.radius})"

    def distance_and_mask(self, p):
        angle_position = normalize_angle(jnp.atan2(p[1], p[0]))

        parametric_position = (angle_position - self.start_angle) / self.angular_length
        mask = clamp_mask(parametric_position, 0, 1)

        parametric_distance = (jnp.linalg.norm(p) - self.radius) * mask

        return parametric_distance, mask


class TranslatedSegment:
    def __init__(self, segment, translation):
        self.segment = segment
        self.translation = translation

    @property
    def start(self):
        return self.segment.start + self.translation

    @property
    def end(self):
        return self.segment.end + self.translation

    @property
    def start_tangent(self):
        return self.segment.start_tangent

    @property
    def end_tangent(self):
        return self.segment.end_tangent

    def __repr__(self):
        return f"TranslatedSegment({self.segment}, {self.translation})"

    def distance_and_mask(self, p):
        return self.segment.distance_and_mask(p - self.translation)


class InversedSegment:
    def __init__(self, segment):
        self.segment = segment

    @property
    def start(self):
        return self.segment.start

    @property
    def end(self):
        return self.segment.end

    @property
    def start_tangent(self):
        return self.segment.start_tangent

    @property
    def end_tangent(self):
        return self.segment.end_tangent

    def __repr__(self):
        return f"InversedSegment({self.segment})"

    def distance_and_mask(self, p):
        distance, mask = self.segment.distance_and_mask(p)
        return -distance, mask


class ClosedPath(Shape):
    def __init__(self, segments):
        self.segments = segments
        self.starts = [segment.start for segment in segments]

        tangents = zip(
            [segment.start_tangent for segment in [segments[-1]] + segments[:-1]],
            [segment.end_tangent for segment in segments],
        )

        self.concave_points = [
            point
            for (tg1, tg2), point in zip(tangents, self.starts)
            if jnp.cross(tg1, tg2) > 0
        ]

        # TODO: Improve this code - some of it should be in the Segment class
        self.orientation_sign = jnp.sign(
            jnp.sum(
                jnp.array(
                    [jnp.cross(segment.start, segment.end) for segment in segments]
                )
            )
        )

    def __repr__(self):
        return (
            f"ClosedPath(\n{"\n".join(indent_shape(seg) for seg in self.segments)}\n)"
        )

    def _min_distance_to_points(self, p):
        dist = jnp.linalg.norm(p - self.starts[0])
        for point in self.starts[1:]:
            dist = jnp.minimum(dist, jnp.linalg.norm(p - point))
        return dist

    def distance(self, p):
        return jax.vmap(self.distance_)(p)

    def distance_(self, p):
        # The gist of the algorithm is to combine the distance to each segment
        # and to each point between the segments. Segments cannot apply to the
        # whole space, so we need to combine them with "masks".

        # We use a mask instead of an if condition as jax jit does not support it

        distances_and_masks = [
            segment.distance_and_mask(p) for segment in self.segments
        ]

        current_distance, current_mask = distances_and_masks[0]
        minimum_distance = jnp.abs(current_distance)

        # We combine the distance for all the segments
        for distance, mask in distances_and_masks[1:]:
            distance = jnp.abs(distance)

            # first we handle the case where the distance can be defined by
            # its distance to both segment
            current_distance = jnp.minimum(
                minimum_distance * mask, distance * current_mask
            )

            # if only one of the segment can apply we use it
            current_distance += minimum_distance * (1 - mask) + distance * (
                1 - current_mask
            )

            current_mask = 1 - ((1 - mask) * (1 - current_mask))
            minimum_distance = current_distance

        points_distance = self._min_distance_to_points(p)
        minimum_distance = (
            jnp.minimum(minimum_distance, points_distance)
            + (1 - current_mask) * points_distance
        )

        current_sign = 1
        # We take the sign from the distance it found
        for distance, _ in distances_and_masks:
            current_sign = jax.lax.select(
                minimum_distance == -distance, -1, current_sign
            )

        for point in self.concave_points:
            point_distance = jnp.linalg.norm(p - point)
            current_sign = jax.lax.select(
                minimum_distance == point_distance, -1, current_sign
            )

        return minimum_distance * current_sign * self.orientation_sign
