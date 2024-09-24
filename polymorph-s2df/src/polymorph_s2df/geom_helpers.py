from polymorph_num.angle import Angle, angle_from_rad
from polymorph_num.expr import PI, Expr
from polymorph_num.vec import Vec2

from polymorph_s2df.paths import BulgingSegment


def bulging_segment_from_start_tangent(
    start_point: Vec2, end_point: Vec2, tangent: Angle
):
    chord = end_point - start_point

    tgt = tangent.as_vec()

    s = chord.cross(tgt)
    c = chord.dot(tgt)

    bulge = -s / (c + (c * c + s * s).sqrt())
    return BulgingSegment(start_point, end_point, bulge)


def bulging_segment_from_end_tangent(
    start_point: Vec2, end_point: Vec2, tangent: Angle
):
    chord = end_point - start_point

    tgt = tangent.as_vec()

    s = tgt.cross(chord)
    c = tgt.dot(chord)

    bulge = -s / (c + (c * c + s * s).sqrt())
    return BulgingSegment(start_point, end_point, bulge)


def line_line_intersection(p0: Vec2, v0: Vec2, p1: Vec2, v1: Vec2):
    cross_dir = v0.cross(v1)
    diff_point = p1 - p0

    param = diff_point.cross(v1) / cross_dir

    return p0 + v0.scale(param)


def biarc(
    x0: Expr, y0: Expr, theta0: Angle, x1: Expr, y1: Expr, theta1: Angle, param: Expr
):
    p0 = Vec2(x0, y0)
    p1 = Vec2(x1, y1)

    u0 = Vec2(theta0.cos(), theta0.sin())
    u1 = Vec2(theta1.cos(), theta1.sin())

    m0 = (p0 + p1) / 2
    v0 = (p1 - p0).perp()

    m1 = (p0 + p1 + u0 + u1) / 2
    v1 = (u1 + p1 - u0 - p0).perp()

    center = line_line_intersection(m0, v0, m1, v1)
    radius = (center - p0).norm()

    mid_angle = (theta0 + theta1).half()
    angle = angle_from_rad((param - 0.5) % 1 * 2 * PI) + mid_angle

    j = center + Vec2(angle.cos(), angle.sin()).scale(radius)

    return [
        bulging_segment_from_start_tangent(p0, j, theta0),
        bulging_segment_from_end_tangent(j, p1, theta1),
    ]
