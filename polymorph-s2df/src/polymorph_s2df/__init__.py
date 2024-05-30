import jax.numpy as jnp
from .shapes import (
    Box,
    Circle,
    TopHalfPlane,
    BottomHalfPlane,
    LeftHalfPlane,
    RightHalfPlane,
)
from .operations import Shape, Intersection, Union, SmoothUnion, SmoothIntersection

from .paths import LineSegment, ClosedPath


def p(x, y):
    return jnp.array([x, y])


def center_and_point_circle(center, point):
    radius = jnp.linalg.norm(center - point)
    return Circle(radius).translate(center)


def two_corners_rectangle(corner1, corner2):
    width = jnp.abs(corner1[0] - corner2[0])
    height = jnp.abs(corner1[1] - corner2[1])
    return Box(width, height).translate((corner1 + corner2) / 2)


def polygon(vertices):
    segments = [
        LineSegment(vertices[i], vertices[(i + 1) % len(vertices)])
        for i in range(len(vertices))
    ]
    return ClosedPath(segments)
