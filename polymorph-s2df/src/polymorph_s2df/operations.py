from .utils import indent_shape

import jax.numpy as jnp
import jax


class Shape:
    def distance(self, p):
        raise NotImplementedError

    def is_inside(self, p, scale=100):
        return 1 - jax.nn.sigmoid(scale * self.distance(p))

    def union(self, other):
        return Union(self, other)

    def intersect(self, other):
        return Intersection(self, other)

    def substract(self, other):
        return Substraction(self, other)

    def smooth_union(self, k, other):
        return SmoothUnion(k, self, other)

    def smooth_intersect(self, k, other):
        return SmoothIntersection(k, self, other)

    def smooth_substract(self, k, other):
        return SmoothSubstraction(k, self, other)

    def shell(self, thickness):
        return Shell(self, thickness)

    def translate(self, offset):
        return Translation(offset, self)

    def rotate(self, angle):
        return Rotation(angle, self)

    def rotate_around(self, angle, center):
        return Translation(center, Rotation(angle, Translation(-center, self)))

    def scale(self, scale):
        return Scale(scale, self)

    def invert(self):
        return Inversion(self)

    def dilate(self, offset):
        return Dilate(offset, self)


class Translation(Shape):
    def __init__(self, offset, shape):
        self.offset = offset
        self.shape = shape

    def __repr__(self):
        return f"Translation(\n  {self.offset},\n{indent_shape(self.shape)}\n)"

    def distance(self, p):
        return self.shape.distance(p - self.offset)


class Rotation(Shape):
    def __init__(self, angle, shape):
        self.angle = angle
        self.shape = shape

        c = jnp.cos(angle)
        s = jnp.sin(angle)
        self.R = jnp.array(
            [
                [c, -s],
                [s, c],
            ]
        )

    def __repr__(self):
        return f"Rotation(\n  {self.angle}, \n{indent_shape(self.shape)}\n)"

    def distance(self, p):
        return self.shape.distance((self.R @ p.T).T)


class Intersection(Shape):
    def __init__(self, shape_1, shape_2):
        self.shape_1 = shape_1
        self.shape_2 = shape_2

    def __repr__(self):
        return f"Intersection(\n{indent_shape(self.shape_1)}, \n{indent_shape(self.shape_2)}\n)"

    def distance(self, p):
        return jnp.maximum(self.shape_1.distance(p), self.shape_2.distance(p))


class SmoothIntersection(Shape):
    def __init__(self, k, shape_1, shape_2):
        self.k = k * 4
        self.shape_1 = shape_1
        self.shape_2 = shape_2

    def __repr__(self):
        return f"SmoothIntersection(\n  {self.k / 4},\n{indent_shape(self.shape_1)}, \n{indent_shape(self.shape_2)} \n)"

    def distance(self, p):
        val1 = self.shape_1.distance(p)
        val2 = self.shape_2.distance(p)

        h = jnp.maximum(self.k - jnp.abs(val1 - val2), 0.0) / self.k
        return jnp.maximum(val1, val2) + h * h * self.k * (1.0 / 4.0)


class Union(Shape):
    def __init__(self, shape_1, shape_2):
        self.shape_1 = shape_1
        self.shape_2 = shape_2

    def __repr__(self):
        return (
            f"Union(\n{indent_shape(self.shape_1)}, \n{indent_shape(self.shape_2)}\n)"
        )

    def distance(self, p):
        return jnp.minimum(self.shape_1.distance(p), self.shape_2.distance(p))


class SmoothUnion(Shape):
    def __init__(self, k, shape_1, shape_2):
        self.k = k * 4
        self.shape_1 = shape_1
        self.shape_2 = shape_2

    def __repr__(self):
        return f"SmoothUnion(\n  {self.k / 4},\n{indent_shape(self.shape_1)}, \n{indent_shape(self.shape_2)} \n)"

    def distance(self, p):
        val1 = self.shape_1.distance(p)
        val2 = self.shape_2.distance(p)

        h = jnp.maximum(self.k - jnp.abs(val1 - val2), 0.0) / self.k
        return jnp.minimum(val1, val2) - h * h * self.k * (1.0 / 4.0)


class Substraction(Shape):
    def __init__(self, shape_1, shape_2):
        self.shape_1 = shape_1
        self.shape_2 = shape_2

    def __repr__(self):
        return (
            f"Union(\n{indent_shape(self.shape_1)}, \n{indent_shape(self.shape_2)}\n)"
        )

    def distance(self, p):
        return jnp.maximum(self.shape_1.distance(p), -self.shape_2.distance(p))


class SmoothSubstraction(Shape):
    def __init__(self, k, shape_1, shape_2):
        self.k = k * 4
        self.shape_1 = shape_1
        self.shape_2 = shape_2

    def __repr__(self):
        return f"SmoothSubstraction(\n  {self.k / 4},\n{indent_shape(self.shape_1)}, \n{indent_shape(self.shape_2)} \n)"

    def distance(self, p):
        val1 = self.shape_1.distance(p)
        val2 = self.shape_2.distance(p)

        h = jnp.maximum(self.k - jnp.abs(val1 + val2), 0.0) / self.k
        return jnp.maximum(val1, -val2) + h * h * self.k * (1.0 / 4.0)


class Scale(Shape):
    def __init__(self, scale, shape):
        self.scale = scale
        self.shape = shape

    def __repr__(self):
        return f"Scale(\n  {self.scale},\n{indent_shape(self.shape)}\n)"

    def distance(self, p):
        return self.shape.distance(p / self.scale) * self.scale


class Inversion(Shape):
    def __init__(self, shape):
        self.shape = shape

    def __repr__(self):
        return f"Inversion(\n{indent_shape(self.shape)}\n)"

    def distance(self, p):
        return -self.shape.distance(p)


class Dilate(Shape):
    def __init__(self, offset, shape):
        self.offset = offset
        self.shape = shape

    def __repr__(self):
        return f"Dilate(\n  {self.offset},\n{indent_shape(self.shape)}\n)"

    def distance(self, p):
        return self.shape.distance(p) - self.offset


class Shell(Shape):
    def __init__(self, shape, thickness):
        self.shape = shape
        self.thickness = thickness

    def __repr__(self):
        return f"Shell(\n  {self.thickness},\n{indent_shape(self.shape)}\n)"

    def distance(self, p):
        return jnp.abs(self.shape.distance(p)) - self.thickness
