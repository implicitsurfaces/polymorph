from .shapes import (
    Box,
    Circle,
    TopHalfPlane,
    BottomHalfPlane,
    LeftHalfPlane,
    RightHalfPlane,
    Triangle,
)
from .operations import Intersection, Union, SmoothUnion, SmoothIntersection

import jax.numpy as jnp


def p(x, y):
    return jnp.array([x, y])


def center_and_point_circle(center, point):
    radius = jnp.linalg.norm(center - point)
    return Circle(radius).translate(center)


def two_corners_rectangle(corner1, corner2):
    width = jnp.abs(corner1[0] - corner2[0])
    height = jnp.abs(corner1[1] - corner2[1])
    return Box(width, height).translate((corner1 + corner2) / 2)
