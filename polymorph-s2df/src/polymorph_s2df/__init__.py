import jax.numpy as jnp
import textwrap
import jax


def _soft_plus(value):
    return jax.nn.softplus(50 * value) / 50


def _soft_minus(value):
    return -jax.nn.softplus(50 * -value) / 50


def _indent_shape(shape):
    return textwrap.indent(repr(shape), "  ")

def p(x,y):
    return jnp.array([x,y])


class Transformable:
    def translate(self, offset):
        return Translation(offset, self)

    def rotate(self, angle):
        return Rotation(angle, self)

    def rotate_around(self, angle, center):
        return Translation(center, Rotation(angle, Translation(-center, self)))


class Modifiable:
    def union(self, other):
        return Union(self, other)

    def intersect(self, other):
        return Intersection(self, other)

    def smooth_intersect(self, k, other):
        return SmoothIntersection(k, self, other)

    def smooth_union(self, k, other):
        return SmoothUnion(k, self, other)


class Shape(Transformable, Modifiable):
    def distance(self, p):
        raise NotImplementedError

    def is_inside(self, p):
        return 1 - jax.nn.sigmoid(100 * self.distance(p))


class HalfPlane(Shape):
    def __init__(self):
        pass

    def __repr__(self):
        return "HalfPlane()"

    def distance(self, p):
        return p[:, 1]


class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def __repr__(self):
        return f"Circle({self.radius})"

    def distance(self, p):
        return jnp.linalg.norm(p) - self.radius


class Box(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __repr__(self):
        return f"Box({self.width}, {self.height})"

    def distance(self, p):
        q = jnp.abs(p) - jnp.array([self.width, self.height]) / 2
        return jnp.linalg.norm(_soft_plus(q)) + _soft_minus(jnp.amax(q))


class Translation(Shape):
    def __init__(self, offset, shape):
        self.offset = offset
        self.shape = shape

    def __repr__(self):
        return f"Translation(\n  {self.offset},\n{_indent_shape(self.shape)}\n)"

    def distance(self, p):
        return self.shape.distance(p - self.offset)


class Rotation(Shape):
    def __init__(self, angle, shape):
        self.angle = angle
        self.shape = shape

        self.cos = jnp.cos(angle)
        self.sin = jnp.sin(angle)

    def __repr__(self):
        return f"Rotation(\n  {self.angle}, \n{_indent_shape(self.shape)}\n)"

    def distance(self, p):
        new_point = jnp.asarray(
            [p[0] * self.cos - p[1] * self.sin, p[0] * self.sin + p[1] * self.cos]
        )

        return self.shape.distance(new_point)


class Intersection(Shape):
    def __init__(self, shape_1, shape_2):
        self.shape_1 = shape_1
        self.shape_2 = shape_2

    def __repr__(self):
        return f"Intersection(\n{_indent_shape(self.shape_1)}, \n{_indent_shape(self.shape_2)}\n)"

    def distance(self, p):
        return jnp.maximum(self.shape_1.distance(p), self.shape_2.distance(p))


class SmoothIntersection(Shape):
    def __init__(self, k, shape_1, shape_2):
        self.k = k * 4
        self.shape_1 = shape_1
        self.shape_2 = shape_2

    def __repr__(self):
        return f"SmoothIntersection(\n  {self.k / 4},\n{_indent_shape(self.shape_1)}, \n{_indent_shape(self.shape_2)} \n)"

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
            f"Union(\n{_indent_shape(self.shape_1)}, \n{_indent_shape(self.shape_2)}\n)"
        )

    def distance(self, p):
        return jnp.minimum(self.shape_1.distance(p), self.shape_2.distance(p))


class SmoothUnion(Shape):
    def __init__(self, k, shape_1, shape_2):
        self.k = k * 4
        self.shape_1 = shape_1
        self.shape_2 = shape_2

    def __repr__(self):
        return f"SmoothUnion(\n  {self.k / 4},\n{_indent_shape(self.shape_1)}, \n{_indent_shape(self.shape_2)} \n)"

    def distance(self, p):
        val1 = self.shape_1.distance(p)
        val2 = self.shape_2.distance(p)

        h = jnp.maximum(self.k - jnp.abs(val1 - val2), 0.0) / self.k
        return jnp.minimum(val1, val2) - h * h * self.k * (1.0 / 4.0)
