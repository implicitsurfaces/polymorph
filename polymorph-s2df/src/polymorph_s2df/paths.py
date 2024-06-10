import jax.numpy as jnp
import jax.lax

from .operations import Shape
from .utils import clamp_mask, indent_shape, repr_point


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
        return f"LineSegment({repr_point(self.start)}, {repr_point(self.end)})"

    def distance_and_mask(self, p):
        start_to_p = p - self.start

        parametric_position = (
            jnp.dot(start_to_p, self.segment) / self.segment_length_square
        )

        mask = clamp_mask(parametric_position, 0, 1)

        projected_point = self.start + self.segment * parametric_position
        return (
            jnp.sign(jnp.cross(start_to_p, self.segment))
            * jnp.linalg.norm(p - projected_point)
            * mask,
            mask,
        )


def normalize_angle(q):
    return jnp.mod(q + 2 * jnp.pi, 2 * jnp.pi)


class ArcSegment:
    def __init__(self, start_angle, end_angle, radius):
        self.start_angle = normalize_angle(start_angle)
        self.end_angle = normalize_angle(end_angle)
        self.radius = radius

        self.start = jnp.array(
            [radius * jnp.cos(self.start_angle), radius * jnp.sin(self.start_angle)]
        )
        self.end = jnp.array([radius * jnp.cos(end_angle), radius * jnp.sin(end_angle)])

        self.angular_length = self.end_angle - self.start_angle

        self.start_tangent = jnp.sign(self.angular_length) * jnp.array(
            [-jnp.sin(self.start_angle), jnp.cos(self.start_angle)]
        )
        self.end_tangent = jnp.sign(self.angular_length) * jnp.array(
            [jnp.sin(self.end_angle), -jnp.cos(self.end_angle)]
        )

    def __repr__(self):
        return f"ArcSegment({self.start_angle}, {self.end_angle}, {self.radius})"

    def distance_and_mask(self, p):
        angle_position = normalize_angle(jnp.atan2(p[1], p[0]))

        parametric_position = (angle_position - self.start_angle) / self.angular_length
        mask = clamp_mask(parametric_position, 0, 1)

        parametric_distance = (jnp.linalg.norm(p) - self.radius) * mask

        return parametric_distance, mask


class QuadraticBezierSegment:
    def __init__(self, start, control, end):
        self.start = start
        self.control = control
        self.end = end

    @property
    def start_tangent(self):
        vector = self.control - self.start
        return vector / jnp.linalg.norm(vector)

    @property
    def end_tangent(self):
        vector = self.control - self.end
        return vector / jnp.linalg.norm(vector)

    def __repr__(self):
        return f"QuadraticBezierSegment({repr_point(self.start)}, {repr_point(self.control)}, {repr_point(self.end)})"

    def distance_and_mask(self, pos):
        a = self.control - self.start
        b = self.start - 2.0 * self.control + self.end
        c = a * 2.0

        d = self.start - pos
        kk = 1.0 / jnp.dot(b, b)
        kx = kk * jnp.dot(a, b)
        ky = kk * (2.0 * jnp.dot(a, a) + jnp.dot(d, b)) / 3.0
        kz = kk * jnp.dot(d, a)

        p = ky - kx * kx
        p3 = p * p * p
        q = kx * (2.0 * kx * kx - 3.0 * ky) + kz
        h = q * q + 4.0 * p3

        def positive_case():
            parametric_position = -kx

            sqrt_h = jnp.sqrt(h)

            for param in (sqrt_h, -sqrt_h):
                x = (param - q) / 2.0
                parametric_position += jnp.sign(x) * jnp.pow(jnp.abs(x), 1.0 / 3.0)

            mask = clamp_mask(parametric_position, 0, 1)
            distance_vector = d + (c + b * parametric_position) * parametric_position
            tangent_vector = c + 2 * b * parametric_position

            sign = jnp.sign(jnp.cross(tangent_vector, distance_vector))

            return sign * jnp.linalg.norm(distance_vector) * mask, mask

        def negative_case():
            z = jnp.sqrt(-p)
            v = jnp.acos(q / (p * z * 2.0)) / 3.0
            m = jnp.cos(v)
            n = jnp.sin(v) * 1.732050808

            parametric_position_1 = (m + m) * z - kx
            parametric_position_2 = (-n - m) * z - kx

            parametric_positions = jnp.array(
                [parametric_position_1, parametric_position_2]
            )

            distance_vectors = jnp.array(
                [
                    d + (c + b * parametric_position_1) * parametric_position_1,
                    d + (c + b * parametric_position_2) * parametric_position_2,
                ]
            )

            distances = jnp.linalg.norm(distance_vectors, axis=1)
            min_index = jnp.argmin(distances)

            mask = clamp_mask(parametric_positions[min_index], 0, 1)
            tangent_vector = c + 2 * b * parametric_positions[min_index]

            sign = jnp.sign(jnp.cross(tangent_vector, distance_vectors[min_index]))

            return sign * distances[min_index] * mask, mask

            # the third root cannot be the closest
            # res = min(res,dot2(d+(c+b*t.z)*t.z))

        return jax.lax.cond(h > 0, positive_case, negative_case)


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
        return f"TranslatedSegment({self.segment}, {repr_point(self.translation)})"

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

        tangents = list(
            zip(
                [segment.start_tangent for segment in segments],
                [segment.end_tangent for segment in segments[-1:] + segments[:-1]],
            )
        )

        self.concave_points = [
            point
            for (tg1, tg2), point in zip(tangents, self.starts)
            if jnp.cross(tg1, tg2) < 0
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
        return f"ClosedPath([\n{',\n'.join(indent_shape(seg) for seg in self.segments)}\n])"

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
