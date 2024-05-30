import jax.numpy as jnp
import jax

from .operations import Shape
from .utils import *


class BottomHalfPlane(Shape):
    def __init__(self):
        pass

    def __repr__(self):
        return "BottomHalfPlane()"

    def distance(self, p):
        return p[:, 1]


class TopHalfPlane(Shape):
    def __init__(self):
        pass

    def __repr__(self):
        return "TopHalfPlane()"

    def distance(self, p):
        return -p[:, 1]


class LeftHalfPlane(Shape):
    def __init__(self):
        pass

    def __repr__(self):
        return "LeftHalfPlane()"

    def distance(self, p):
        return p[:, 0]


class RightHalfPlane(Shape):
    def __init__(self):
        pass

    def __repr__(self):
        return "RightHalfPlane()"

    def distance(self, p):
        return -p[:, 0]


class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def __repr__(self):
        return f"Circle({self.radius})"

    def distance(self, p):
        return length(p) - self.radius


class Box(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.center = jnp.array([self.width, self.height]) / 2

    def __repr__(self):
        return f"Box({self.width}, {self.height})"

    def distance(self, p):
        q = jnp.abs(p) - self.center
        return length(soft_plus(q)) + soft_minus(jnp.amax(q, axis=1))


class Triangle(Shape):
    def __init__(self, p1, p2, p3):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

        self.e0 = self.p2 - self.p1
        self.e1 = self.p3 - self.p2
        self.e2 = self.p1 - self.p3

        self.s = jnp.sign(self.e0[0] * self.e2[1] - self.e0[1] * self.e2[0])

    def __repr__(self):
        return f"Triangle({self.p1}, {self.p2}, {self.p3})"

    def distance(self, p):
        # TODO: sorry Steve, I couldn't figure out how to vectorize your distance code. IIRC it's still WIP.
        return jnp.array([self.distance_unvectorized(point) for point in p])

    def distance_unvectorized(self, p):
        v0 = p - self.p1
        v1 = p - self.p2
        v2 = p - self.p3

        pq0 = v0 - self.e0 * jnp.clip(
            jnp.dot(v0, self.e0) / jnp.dot(self.e0, self.e0), 0, 1
        )
        pq0_dist = jnp.dot(pq0, pq0)
        pq0_sign = v0[0] * self.e0[1] - v0[1] * self.e0[0]

        pq1 = v1 - self.e1 * jnp.clip(
            jnp.dot(v1, self.e1) / jnp.dot(self.e1, self.e1), 0, 1
        )
        pq1_dist = jnp.dot(pq1, pq1)
        pq1_sign = v1[0] * self.e1[1] - v1[1] * self.e1[0]

        pq2 = v2 - self.e2 * jnp.clip(
            jnp.dot(v2, self.e2) / jnp.dot(self.e2, self.e2), 0, 1
        )
        pq2_dist = jnp.dot(pq2, pq2)
        pq2_sign = v2[0] * self.e2[1] - v2[1] * self.e2[0]

        dist_array = jnp.array([pq0_dist, pq1_dist, pq2_dist])
        sign_array = jnp.array([pq0_sign, pq1_sign, pq2_sign])

        min_arg = jnp.argmin(dist_array)
        min_dist = dist_array[min_arg]
        min_sign = jnp.sign(self.s * sign_array[min_arg])

        return -min_sign * jnp.sqrt(min_dist)
