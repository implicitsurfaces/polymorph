from polymorph_num import ops
from polymorph_num.expr import Num, as_expr
from polymorph_num.vec3 import Vec3

from polymorph_s2df.solid_operations import Solid

from .operations import Shape
from .plane import XY_PLANE, Plane
from .utils import norm


class ExtrudedShape(Solid):
    def __init__(self, shape: Shape, base: Plane, depth: Num):
        self.shape = shape
        self.depth = as_expr(depth)
        self.base = base

    def distance(self, x: Num, y: Num, z: Num):
        # We project the point in the base plane coordinates
        p = self.base.local_coordinates(Vec3(x, y, z))

        xy_distance = self.shape.distance(p.x, p.y)
        z_distance = p.z.abs() - self.depth

        # This is non zero only  when both distance are negative (i.e. inside)
        # In that case we choose the closest distance (the max of two negative numbers)
        inside_distance = ops.min(ops.max(xy_distance, z_distance), 0)

        # this is non zero only when at least one distance is positive (i.e. outside)
        # In that case we take the norm of the two distances as xy_dist is (x^2 + y^2)^0.5
        # norm(xy_distance, z_distance) is (x^2 + y^2 + z^2)^0.5
        xy_outside_distance = ops.max(xy_distance, 0)
        z_outside_distance = ops.max(z_distance, 0)
        outside_distance = norm(xy_outside_distance, z_outside_distance)

        return inside_distance + outside_distance


class EmbeddedShape:
    def __init__(self, shape: Shape, plane: Plane = XY_PLANE):
        self.shape = shape
        self.plane = plane

    def extrude(self, depth: Num):
        return ExtrudedShape(self.shape, self.plane, depth / 2)
