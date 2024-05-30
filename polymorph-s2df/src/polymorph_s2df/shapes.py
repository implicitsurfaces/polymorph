import jax.numpy as jnp

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
