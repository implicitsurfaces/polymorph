from collections.abc import Callable

from polymorph_num import ops
from polymorph_num.expr import TAU, ZERO, Expr, Num, as_expr
from polymorph_num.vec3 import Vec3

from polymorph_s2df.solid_operations import Solid

from .operations import Shape
from .plane import XY_PLANE, Plane
from .utils import (
    angular_distance,
    max_non_zero,
    min_non_zero,
    norm,
    normalize_angle,
)


class RevolvedShape(Solid):
    def __init__(self, shape: Shape, plane: Plane):
        self.shape = shape
        self.plane = plane

    def distance(self, x: Num, y: Num, z: Num):
        # We project the point in the base plane coordinates
        p = self.plane.local_coordinates(Vec3(x, y, z))

        # We rotate around the y axis. This means that we have the full rotational
        # symmetry around the y axis. The value we are interested in for the base
        # plane is the radius of the xz point
        r = norm(p.x, p.z)
        return self.shape.distance(r, p.y)


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


class ModulatedExtrusion(Solid):
    def __init__(
        self,
        modulationFunction: Callable[[Expr, Expr, Expr], Expr],
        base: Plane,
        depth: Num,
    ):
        self.modulationFunction = modulationFunction
        self.depth = as_expr(depth)
        self.base = base

    def distance(self, x: Num, y: Num, z: Num):
        # We project the point in the base plane coordinates
        p = self.base.local_coordinates(Vec3(x, y, z))

        z_distance = p.z.abs() - self.depth

        t = ops.clamp(p.z / (2 * self.depth) + 0.5, 0, 1)
        xy_distance = self.modulationFunction(p.x, p.y, t)

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


class ArcExtrusion(Solid):
    def __init__(self, shape: Shape, plane: Plane, angle: Num, radius: Num = 0):
        self.shape = shape
        self.plane = plane
        self.angle = as_expr(angle)
        self.radius = as_expr(radius)
        self.orientation_sign = 1

    def distance(self, x: Num, y: Num, z: Num):
        # We want to be centered on the plane - but we will rotate around the radius
        translation = self.plane.xAxis.scale(-self.radius)
        coords = Vec3(x, y, z).translateTo(translation)

        # We project the point in the base plane coordinates
        p = self.plane.local_coordinates(coords)

        # The angle is between 0 and 2 PI (TAU)
        angle_position = normalize_angle(ops.atan2(p.z, p.x))

        start_xy_distance = self.shape.distance(p.x - self.radius, p.y)
        start_z_distance = -p.z

        # For the inside distance we only care about the z distance (the sides
        # are covered by the borders of the shape)
        # To check that a point is inside the shape we need to check that the
        # both distances are inside the shape
        start_same_side = ((start_xy_distance * start_z_distance).sign() + 1) / 2
        start_inside_distance = start_same_side * ops.min(start_z_distance, 0)

        # For the outside distance, we do not care on which (z) side we are on
        # from the shape.
        start_outside_distance = norm(
            ops.max(start_xy_distance, 0), start_z_distance.abs()
        )

        # For the end, we do the same, but on the rotated plane
        rotated_plane = self.plane.pivot(-self.angle, self.plane.yAxis)
        translation = rotated_plane.xAxis.scale(self.radius)
        rotated_plane = rotated_plane.translateTo(translation)
        rotated_p = rotated_plane.local_coordinates(coords)

        end_xy_distance = self.shape.distance(rotated_p.x, rotated_p.y)
        end_z_distance = rotated_p.z

        end_same_side = ((end_xy_distance * end_z_distance).sign() + 1) / 2
        end_inside_distance = end_same_side * ops.min(end_z_distance, 0)
        end_outside_distance = norm(ops.max(end_xy_distance, 0), end_z_distance.abs())

        # The parametric position, between 0 and 1, of the point on the arc
        parametric_position = (
            angular_distance(0, angle_position, self.orientation_sign) / self.angle
        )

        param_bigger_than_0 = ops.if_ge(parametric_position, 0, 1, 0)
        param_smaller_than_1 = ops.if_lt(parametric_position, 1, 1, 0)
        in_extrusion = param_bigger_than_0 * param_smaller_than_1

        # We compute the distance to the full revolution, but only use it if we are within the 0, 1 parameter range
        distance_to_extruded = (
            self.shape.distance(norm(p.x, p.z) - self.radius, p.y) * in_extrusion
        )

        caps_inside_distance = max_non_zero(start_inside_distance, end_inside_distance)
        inside_extruded_distance = ops.min(distance_to_extruded, 0)
        inside_distance = ops.if_lt(
            caps_inside_distance,
            0,
            ops.max(caps_inside_distance, inside_extruded_distance),
            inside_extruded_distance,
        )

        caps_outside_distance = min_non_zero(
            start_outside_distance, end_outside_distance
        )
        outside_distance = (1 - in_extrusion) * caps_outside_distance + ops.max(
            distance_to_extruded, 0
        )

        return inside_distance + outside_distance


class ModulatedArcExtrusion(Solid):
    def __init__(
        self,
        modulationFunction: Callable[[Expr, Expr, Expr], Expr],
        plane: Plane,
        angle: Num,
        radius: Num = 0,
    ):
        self.modulationFunction = modulationFunction
        self.angle = as_expr(angle)
        self.radius = as_expr(radius)
        self.plane = plane

        self.orientation_sign = 1

    def distance(self, x: Num, y: Num, z: Num):
        # We want to be centered on the plane - but we will rotate around the radius
        translation = self.plane.xAxis.scale(-self.radius)
        coords = Vec3(x, y, z).translateTo(translation)

        # We project the point in the base plane coordinates
        p = self.plane.local_coordinates(coords)

        # The angle is between 0 and 2 PI (TAU)
        angle_position = normalize_angle(ops.atan2(p.z, p.x))

        start_xy_distance = self.modulationFunction(p.x - self.radius, p.y, ZERO)
        start_z_distance = -p.z

        # For the inside distance we only care about the z distance (the sides
        # are covered by the borders of the shape)
        # To check that a point is inside the shape we need to check that the
        # both distances are inside the shape
        start_same_side = ((start_xy_distance * start_z_distance).sign() + 1) / 2
        start_inside_distance = start_same_side * ops.min(start_z_distance, 0)

        # For the outside distance, we do not care on which (z) side we are on
        # from the shape.
        start_outside_distance = norm(
            ops.max(start_xy_distance, 0), start_z_distance.abs()
        )

        # For the end, we do the same, but on the rotated plane
        rotated_plane = self.plane.pivot(-self.angle, self.plane.yAxis)
        translation = rotated_plane.xAxis.scale(self.radius)
        rotated_plane = rotated_plane.translateTo(translation)
        rotated_p = rotated_plane.local_coordinates(coords)

        end_xy_distance = self.modulationFunction(rotated_p.x, rotated_p.y, as_expr(1))
        end_z_distance = rotated_p.z

        end_same_side = ((end_xy_distance * end_z_distance).sign() + 1) / 2
        end_inside_distance = end_same_side * ops.min(end_z_distance, 0)
        end_outside_distance = norm(ops.max(end_xy_distance, 0), end_z_distance.abs())

        # The parametric position, between 0 and 1, of the point on the arc
        parametric_position = (
            angular_distance(0, angle_position, self.orientation_sign) / self.angle
        )

        param_bigger_than_0 = ops.if_ge(parametric_position, 0, 1, 0)
        param_smaller_than_1 = ops.if_lt(parametric_position, 1, 1, 0)
        in_extrusion = param_bigger_than_0 * param_smaller_than_1

        # We compute the distance to the full revolution, but only use it if we are within the 0, 1 parameter range
        distance_to_extruded = (
            self.modulationFunction(
                norm(p.x, p.z) - self.radius, p.y, ops.clamp(parametric_position, 0, 1)
            )
            * in_extrusion
        )

        caps_inside_distance = max_non_zero(start_inside_distance, end_inside_distance)
        inside_extruded_distance = ops.min(distance_to_extruded, 0)
        inside_distance = ops.if_lt(
            caps_inside_distance,
            0,
            ops.max(caps_inside_distance, inside_extruded_distance),
            inside_extruded_distance,
        )

        caps_outside_distance = min_non_zero(
            start_outside_distance, end_outside_distance
        )
        outside_distance = (1 - in_extrusion) * caps_outside_distance + ops.max(
            distance_to_extruded, 0
        )

        return inside_distance + outside_distance


class EmbeddedShape:
    def __init__(self, shape: Shape, plane: Plane = XY_PLANE):
        self.shape = shape
        self.plane = plane

    def revolve(self):
        return RevolvedShape(self.shape, self.plane)

    def extrude(self, depth: Num):
        return ExtrudedShape(self.shape, self.plane, depth / 2)

    def morph_extrude(self, end_shape: Shape, depth: Num):
        def morph_modulation(x: Expr, y: Expr, t: Expr):
            return self.shape.morph(t, end_shape).distance(x, y)

        return ModulatedExtrusion(morph_modulation, self.plane, depth / 2)

    def twist_extrude(self, depth: Num, twist_angle: Num):
        def twist_modulation(x: Expr, y: Expr, t: Expr):
            return self.shape.rotate(twist_angle * TAU * t).distance(x, y)

        return ModulatedExtrusion(twist_modulation, self.plane, depth / 2)

    def arc_extrude(self, angle: Num, radius: Num = 0):
        return ArcExtrusion(self.shape, self.plane, angle, radius)

    def morph_arc_extrude(self, end_shape: Shape, angle: Num, radius: Num = 0):
        def morph_modulation(x: Expr, y: Expr, t: Expr):
            return self.shape.morph(t, end_shape).distance(x, y)

        return ModulatedArcExtrusion(morph_modulation, self.plane, angle, radius)

    def twist_arc_extrude(self, angle: Num, twist_angle: Num, radius: Num = 0):
        def twist_modulation(x: Expr, y: Expr, t: Expr):
            return self.shape.rotate(twist_angle * TAU * t).distance(x, y)

        return ModulatedArcExtrusion(twist_modulation, self.plane, angle, radius)
