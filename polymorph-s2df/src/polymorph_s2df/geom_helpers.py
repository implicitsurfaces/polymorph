from polymorph_num import ops
from polymorph_num.expr import PI, Expr, Num, as_expr
from polymorph_num.vec import ValVec, Vec2, as_vec2

from polymorph_s2df.paths import ArcSegment, TranslatedSegment


def bulge_arc(point1: ValVec, point2: ValVec, bulge: Num):
    point1 = as_vec2(point1)
    point2 = as_vec2(point2)
    bulge = as_expr(bulge)

    half_chord = (point2 - point1).norm() / 2

    # the sagitta is the perpendicular distance from the midpoint of the chord to the arc
    sagitta: Expr = bulge.abs() * half_chord

    midpoint = (point1 + point2) / 2

    radius = (half_chord * half_chord + sagitta * sagitta) / (sagitta * 2)

    # Calculate the direction vector perpendicular to the chord
    direction = Vec2(point2.y - point1.y, point1.x - point2.x)
    direction = direction / direction.norm()

    center = midpoint + direction.scale((radius - sagitta) * bulge.sign())

    # Calculate the angles
    diff1 = point1 - center
    angle1 = ops.atan2(diff1.y, diff1.x)
    diff2 = point2 - center
    angle2 = ops.atan2(diff2.y, diff2.x)

    return TranslatedSegment(ArcSegment(angle1, angle2, radius, -bulge.sign()), center)


def extrema_points_and_tangent(
    start_point: Vec2, end_point: Vec2, tangent: Expr, reverse: bool = False
):
    chord_direction = Vec2(end_point.y - start_point.y, start_point.x - end_point.x)
    chord_midpoint = (start_point + end_point) / 2

    tangent_vector = Vec2(tangent.cos(), tangent.sin())
    # We do not care about the orientation here - as we are interested in the direction
    # in order to compute the intersection point
    tgt_normal = tangent_vector.perp()

    center = line_line_intersection(
        start_point, tgt_normal, chord_midpoint, chord_direction
    )
    radius = (start_point - center).norm()

    # Calculate the angles
    diff = end_point - center
    angle2 = ops.atan2(diff.y, diff.x)

    # This is not a unit vector, but we are only interested in the direction
    normal_vector = start_point - center

    orientation_sign = normal_vector.cross(tangent_vector).sign()
    normal_angle = tangent - orientation_sign * PI / 2

    start_angle = normal_angle if not reverse else angle2
    end_angle = angle2 if not reverse else normal_angle

    orientation_sign = (-1 if reverse else 1) * orientation_sign

    arc = ArcSegment(start_angle, end_angle, radius, orientation_sign)

    return TranslatedSegment(arc, center)


def line_line_intersection(p0: Vec2, v0: Vec2, p1: Vec2, v1: Vec2):
    cross_dir = v0.cross(v1)
    diff_point = p1 - p0

    param = diff_point.cross(v1) / cross_dir

    return p0 + v0.scale(param)


def biarc(
    x0: Expr, y0: Expr, theta0: Expr, x1: Expr, y1: Expr, theta1: Expr, param: Expr
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

    mid_angle = (theta0 + theta1) / 2
    angle = (param - 0.5) % 1 * 2 * PI + mid_angle

    j = center + Vec2(angle.cos(), angle.sin()).scale(radius)

    return [
        extrema_points_and_tangent(p0, j, theta0),
        extrema_points_and_tangent(p1, j, PI + theta1, reverse=True),
    ]
