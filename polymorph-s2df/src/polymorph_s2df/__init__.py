import jax.numpy as jnp

# Note: `X as X` stops ruff (and other tools) from flagging imports as unused.
# See https://docs.astral.sh/ruff/rules/unused-import/
from .operations import Intersection as Intersection
from .operations import Shape as Shape
from .operations import SmoothIntersection as SmoothIntersection
from .operations import SmoothUnion as SmoothUnion
from .operations import Union as Union
from .paths import ArcSegment as ArcSegment
from .paths import ClosedPath as ClosedPath
from .paths import InversedSegment as InversedSegment
from .paths import LineSegment as LineSegment
from .paths import QuadraticBezierSegment as QuadraticBezierSegment
from .paths import TranslatedSegment as TranslatedSegment
from .shapes import BottomHalfPlane as BottomHalfPlane
from .shapes import Box as Box
from .shapes import Circle as Circle
from .shapes import LeftHalfPlane as LeftHalfPlane
from .shapes import RightHalfPlane as RightHalfPlane
from .shapes import TopHalfPlane as TopHalfPlane


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


def bulge_arc(point1, point2, bulge):
    chord_length = jnp.linalg.norm(point2 - point1)

    # the sagitta is the perpendicular distance from the midpoint of the chord to the arc
    sagitta = (jnp.abs(bulge) * chord_length) / 2

    midpoint = (point1 + point2) / 2

    radius = ((chord_length / 2) ** 2 + sagitta**2) / (2 * sagitta)

    # Calculate the direction vector perpendicular to the chord
    direction = jnp.array([point2[1] - point1[1], point1[0] - point2[0]])
    direction = direction / jnp.linalg.norm(direction)

    if bulge > 0:
        center = midpoint + direction * (radius - sagitta)
    else:
        center = midpoint - direction * (radius - sagitta)

    # Calculate the angles
    angle1 = jnp.arctan2(point1[1] - center[1], point1[0] - center[0])
    angle2 = jnp.arctan2(point2[1] - center[1], point2[0] - center[0])

    segment = TranslatedSegment(ArcSegment(angle1, angle2, radius), center)
    return segment if bulge < 0 else InversedSegment(segment)


def bulging_polygon(points):
    previous_point = points[0]
    previous_bulge = 0

    segments = []

    for point_or_bulge in points[1:] + [points[0]]:
        if jnp.isscalar(point_or_bulge):
            previous_bulge = point_or_bulge
        else:
            segment = (
                bulge_arc(previous_point, point_or_bulge, previous_bulge)
                if previous_bulge != 0
                else LineSegment(previous_point, point_or_bulge)
            )
            segments.append(segment)
            previous_point = point_or_bulge
            previous_bulge = 0

    return ClosedPath(segments)
