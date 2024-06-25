import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from .utils import indent_shape


class Shape:
    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(*children)

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
        return Shell(thickness, self)

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

    def morph(self, t, other):
        return Morph(t, self, other)

    def taper(self, height, scale):
        return Taper(height, scale, self)

    def taper_from_point(self, point, height, scale):
        return Translation(point, Taper(height, scale, Translation(-point, self)))


@register_pytree_node_class
class Translation(Shape):
    def __init__(self, offset, shape):
        self.offset = offset
        self.shape = shape

    def __repr__(self):
        return f"Translation(\n  {self.offset},\n{indent_shape(self.shape)}\n)"

    def tree_flatten(self):
        return (self.offset, self.shape), None

    def distance(self, p):
        return self.shape.distance(p - self.offset)


@register_pytree_node_class
class Rotation(Shape):
    def __init__(self, angle, shape):
        self.angle = angle
        self.shape = shape

    @property
    def R(self):
        c = jnp.cos(self.angle)
        s = jnp.sin(self.angle)
        return jnp.array(
            [
                [c, -s],
                [s, c],
            ]
        )

    def __repr__(self):
        return f"Rotation(\n  {self.angle}, \n{indent_shape(self.shape)}\n)"

    def tree_flatten(self):
        return (self.angle, self.shape), None

    def distance(self, p):
        return self.shape.distance((self.R @ p.T).T)


@register_pytree_node_class
class Intersection(Shape):
    def __init__(self, shape_1, shape_2):
        self.shape_1 = shape_1
        self.shape_2 = shape_2

    def __repr__(self):
        return f"Intersection(\n{indent_shape(self.shape_1)}, \n{indent_shape(self.shape_2)}\n)"

    def tree_flatten(self):
        return (self.shape_1, self.shape_2), None

    def distance(self, p):
        return jnp.maximum(self.shape_1.distance(p), self.shape_2.distance(p))


@register_pytree_node_class
class SmoothIntersection(Shape):
    def __init__(self, k, shape_1, shape_2):
        self.k = k * 4
        self.shape_1 = shape_1
        self.shape_2 = shape_2

    def __repr__(self):
        return f"SmoothIntersection(\n  {self.k / 4},\n{indent_shape(self.shape_1)}, \n{indent_shape(self.shape_2)} \n)"

    def tree_flatten(self):
        return (self.k / 4, self.shape_1, self.shape_2), None

    def distance(self, p):
        val1 = self.shape_1.distance(p)
        val2 = self.shape_2.distance(p)

        h = jnp.maximum(self.k - jnp.abs(val1 - val2), 0.0) / self.k
        return jnp.maximum(val1, val2) + h * h * self.k * (1.0 / 4.0)


@register_pytree_node_class
class Union(Shape):
    def __init__(self, shape_1, shape_2):
        self.shape_1 = shape_1
        self.shape_2 = shape_2

    def __repr__(self):
        return (
            f"Union(\n{indent_shape(self.shape_1)}, \n{indent_shape(self.shape_2)}\n)"
        )

    def tree_flatten(self):
        return (self.shape_1, self.shape_2), None

    def distance(self, p):
        return jnp.minimum(self.shape_1.distance(p), self.shape_2.distance(p))


@register_pytree_node_class
class SmoothUnion(Shape):
    def __init__(self, k, shape_1, shape_2):
        self.k = k * 4
        self.shape_1 = shape_1
        self.shape_2 = shape_2

    def __repr__(self):
        return f"SmoothUnion(\n  {self.k / 4},\n{indent_shape(self.shape_1)}, \n{indent_shape(self.shape_2)} \n)"

    def tree_flatten(self):
        return (self.k / 4, self.shape_1, self.shape_2), None

    def distance(self, p):
        val1 = self.shape_1.distance(p)
        val2 = self.shape_2.distance(p)

        h = jnp.maximum(self.k - jnp.abs(val1 - val2), 0.0) / self.k
        return jnp.minimum(val1, val2) - h * h * self.k * (1.0 / 4.0)


@register_pytree_node_class
class Substraction(Shape):
    def __init__(self, shape_1, shape_2):
        self.shape_1 = shape_1
        self.shape_2 = shape_2

    def __repr__(self):
        return f"Substraction(\n{indent_shape(self.shape_1)}, \n{indent_shape(self.shape_2)}\n)"

    def tree_flatten(self):
        return (self.shape_1, self.shape_2), None

    def distance(self, p):
        return jnp.maximum(self.shape_1.distance(p), -self.shape_2.distance(p))


@register_pytree_node_class
class SmoothSubstraction(Shape):
    def __init__(self, k, shape_1, shape_2):
        self.k = k * 4
        self.shape_1 = shape_1
        self.shape_2 = shape_2

    def __repr__(self):
        return f"SmoothSubstraction(\n  {self.k / 4},\n{indent_shape(self.shape_1)}, \n{indent_shape(self.shape_2)} \n)"

    def tree_flatten(self):
        return (self.k / 4, self.shape_1, self.shape_2), None

    def distance(self, p):
        val1 = self.shape_1.distance(p)
        val2 = self.shape_2.distance(p)

        h = jnp.maximum(self.k - jnp.abs(val1 + val2), 0.0) / self.k
        return jnp.maximum(val1, -val2) + h * h * self.k * (1.0 / 4.0)


@register_pytree_node_class
class Scale(Shape):
    def __init__(self, scale, shape):
        self.scale = scale
        self.shape = shape

    def __repr__(self):
        return f"Scale(\n  {self.scale},\n{indent_shape(self.shape)}\n)"

    def tree_flatten(self):
        return (self.scale, self.shape), None

    def distance(self, p):
        return self.shape.distance(p / self.scale) * self.scale


@register_pytree_node_class
class Inversion(Shape):
    def __init__(self, shape):
        self.shape = shape

    def __repr__(self):
        return f"Inversion(\n{indent_shape(self.shape)}\n)"

    def tree_flatten(self):
        return (self.shape,), None

    def distance(self, p):
        return -self.shape.distance(p)


@register_pytree_node_class
class Dilate(Shape):
    def __init__(self, offset, shape):
        self.offset = offset
        self.shape = shape

    def __repr__(self):
        return f"Dilate(\n  {self.offset},\n{indent_shape(self.shape)}\n)"

    def tree_flatten(self):
        return (self.offset, self.shape), None

    def distance(self, p):
        return self.shape.distance(p) - self.offset


@register_pytree_node_class
class Shell(Shape):
    def __init__(self, thickness, shape):
        self.shape = shape
        self.thickness = thickness

    def __repr__(self):
        return f"Shell(\n  {self.thickness},\n{indent_shape(self.shape)}\n)"

    def distance(self, p):
        return jnp.abs(self.shape.distance(p)) - self.thickness

    def tree_flatten(self):
        return (self.thickness, self.shape), None


@register_pytree_node_class
class Morph(Shape):
    def __init__(self, t, shape_1, shape_2):
        self.shape_1 = shape_1
        self.shape_2 = shape_2
        self.t = t

    def __repr__(self):
        return f"Morph(\n  {self.t},\n{indent_shape(self.shape_1)}, \n{indent_shape(self.shape_2)}\n)"

    def tree_flatten(self):
        return (self.t, self.shape_1, self.shape_2), None

    def distance(self, p):
        return (1 - self.t) * self.shape_1.distance(p) + self.t * self.shape_2.distance(
            p
        )


@register_pytree_node_class
class Taper(Shape):
    def __init__(self, height, scale, shape):
        self.height = height
        self.scale = scale
        self.shape = shape

    def __repr__(self):
        return f"Taper(\n  {self.height}, {self.scale}, \n{indent_shape(self.shape)}\n)"

    def tree_flatten(self):
        return (self.height, self.scale, self.shape), None

    def distance(self, p):
        s = self.height / (self.scale * p[:, 1] + (self.height - p[:, 1]))
        print(s)
        updated_p = p.at[:, 0].set(p[:, 0] * s)
        return self.shape.distance(updated_p)
