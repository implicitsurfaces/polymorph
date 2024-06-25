import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from .operations import Shape
from .utils import length, soft_minus, soft_plus


@register_pytree_node_class
class BottomHalfPlane(Shape):
    def __init__(self):
        pass

    def __repr__(self):
        return "BottomHalfPlane()"

    def tree_flatten(self):
        return (), None

    def distance(self, p):
        return p[:, 1]


@register_pytree_node_class
class TopHalfPlane(Shape):
    def __init__(self):
        pass

    def __repr__(self):
        return "TopHalfPlane()"

    def tree_flatten(self):
        return (), None

    def distance(self, p):
        return -p[:, 1]


@register_pytree_node_class
class LeftHalfPlane(Shape):
    def __init__(self):
        pass

    def __repr__(self):
        return "LeftHalfPlane()"

    def tree_flatten(self):
        return (), None

    def distance(self, p):
        return p[:, 0]


@register_pytree_node_class
class RightHalfPlane(Shape):
    def __init__(self):
        pass

    def __repr__(self):
        return "RightHalfPlane()"

    def tree_flatten(self):
        return (), None

    def distance(self, p):
        return -p[:, 0]


@register_pytree_node_class
class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def __repr__(self):
        return f"Circle({self.radius})"

    def tree_flatten(self):
        return (self.radius,), None

    def distance(self, p):
        return length(p) - self.radius


@register_pytree_node_class
class Box(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    @property
    def center(self):
        return jnp.array([self.width, self.height]) / 2

    def __repr__(self):
        return f"Box({self.width}, {self.height})"

    def tree_flatten(self):
        return (self.width, self.height), None

    def distance(self, p):
        q = jnp.abs(p) - self.center
        return length(soft_plus(q)) + soft_minus(jnp.amax(q, axis=1))
