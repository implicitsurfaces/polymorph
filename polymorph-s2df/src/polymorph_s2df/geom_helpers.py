from polymorph_num.angle import Angle, polar_angle_from_vec
from polymorph_num.expr import Expr
from polymorph_num.vec import Vec2

from polymorph_s2df.paths import BulgingSegment, LineSegment


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


def bulging_segment_from_center(start_point: Vec2, end_point: Vec2, center: Vec2):
    chord = end_point - start_point
    radius = center - start_point

    s = radius.cross(chord)
    c = radius.dot(chord)

    bulge = -s / (c + (c * c + s * s).sqrt())
    return BulgingSegment(start_point, end_point, bulge)


def bulging_segement_from_radius_small_arc(
    start_point: Vec2, end_point: Vec2, radius: Expr
):
    d = (end_point - start_point).norm()
    f = (d / (2 * radius)) ** 2
    bulge = (2 * radius / d) * (1 - (1 - f * f).sqrt())
    return BulgingSegment(start_point, end_point, bulge)


def bulging_segement_from_radius_large_arc(
    start_point: Vec2, end_point: Vec2, radius: Expr
):
    d = (end_point - start_point).norm()
    f = (d / (2 * radius)) ** 2
    bulge = (2 * radius / d) * (1 + (1 - f * f).sqrt())
    return BulgingSegment(start_point, end_point, bulge)


def line_line_intersection(p0: Vec2, v0: Vec2, p1: Vec2, v1: Vec2):
    cross_dir = v0.cross(v1)
    diff_point = p1 - p0

    param = diff_point.cross(v1) / cross_dir

    return p0 + v0.scale(param)


def _biarc_point_no_inflexion_point(p0: Vec2, theta0: Angle, p1: Vec2, theta1: Angle):
    """
    Returns the point of junciton of the two arcs of a biarc.

    With this algorithm, we assume a biarc that does not have an inflexion
    point. Unfortunately, this is not alway possible and it returns weird
    results when outside of its domain of validity (i.e. when p0 and p1 are in
    the first and fouth quadrant, but not in the same one.)
    """

    third_point = line_line_intersection(p0, theta0.as_vec(), p1, theta1.as_vec())

    w0 = (third_point - p1).norm()
    w1 = (third_point - p0).norm()
    w2 = (p1 - p0).norm()

    perimeter = w0 + w1 + w2

    return (p0.scale(w0) + p1.scale(w1) + third_point.scale(w2)) / perimeter


def reflect_vector(p, v):
    v = v / v.norm()
    projection = v.scale(p.dot(v))
    return 2 * projection - p


def _biarc_point_inflexion_point(p0: Vec2, theta0: Angle, p1: Vec2, theta1: Angle):
    """
    Returns the point of junciton of the two arcs of a biarc.

    With this algorithm, the biarc always has an inflexion point. We choose the
    point such that its tangent is the average of the two tangents, reflected
    throught the chord.

    This corresponds to the algorithm proposed by Bolton (https://doi.org/10.1016/0010-4485(75)90086-X)
    """

    mid_tangent = (theta0.as_vec() + theta1.as_vec()) / 2

    chord = p1 - p0
    tgt = reflect_vector(mid_tangent, chord / chord.norm())
    tgt = polar_angle_from_vec(tgt)

    sweep0 = (theta0 + tgt).half()
    sweep1 = (theta1 + tgt).half()

    return line_line_intersection(p0, sweep0.as_vec(), p1, sweep1.as_vec())


def _biarc_locus_center(p0: Vec2, theta0: Angle, p1: Vec2, theta1: Angle):
    """
    Find the center of the locus circle of the biarc.

    This is the circle that defines all the valid junction points of the biarc.
    """
    u0 = theta0.as_vec()
    u1 = theta1.as_vec()

    m0 = (p0 + p1) / 2
    v0 = (p1 - p0).perp()

    m1 = (p0 + p1 + u0 + u1) / 2
    v1 = (u1 + p1 - u0 - p0).perp()

    return line_line_intersection(m0, v0, m1, v1)


def simple_biarc(x0: Expr, y0: Expr, theta0: Angle, x1: Expr, y1: Expr, theta1: Angle):
    p0 = Vec2(x0, y0)
    p1 = Vec2(x1, y1)

    c = _biarc_point_inflexion_point(p0, theta0, p1, theta1)

    return [
        bulging_segment_from_start_tangent(p0, c, theta0),
        bulging_segment_from_end_tangent(c, p1, theta1),
    ]


def no_inflexion_biarc(
    x0: Expr, y0: Expr, theta0: Angle, x1: Expr, y1: Expr, theta1: Angle
):
    p0 = Vec2(x0, y0)
    p1 = Vec2(x1, y1)

    c = _biarc_point_no_inflexion_point(p0, theta0, p1, theta1)

    return [
        bulging_segment_from_start_tangent(p0, c, theta0),
        bulging_segment_from_end_tangent(c, p1, theta1),
    ]


def biarc(
    x0: Expr, y0: Expr, theta0: Angle, x1: Expr, y1: Expr, theta1: Angle, param: Angle
):
    p0 = Vec2(x0, y0)
    p1 = Vec2(x1, y1)

    c = _biarc_point_inflexion_point(p0, theta0, p1, theta1)

    locus_center = _biarc_locus_center(p0, theta0, p1, theta1)
    radius = (locus_center - p0).norm()

    c_angle = polar_angle_from_vec(c - locus_center)
    angle = param + c_angle

    j = locus_center + angle.as_vec().scale(radius)

    return [
        bulging_segment_from_start_tangent(p0, j, theta0),
        bulging_segment_from_end_tangent(j, p1, theta1),
    ]


def fillet_line_line(line1: LineSegment, line2: LineSegment, radius: Expr):
    corner = line1.end
    ccw = (line2.end_tangent() - line1.end_tangent()).sin().sign()

    mid_tangent = (line1.end_tangent().as_vec() + line2.start_tangent().as_vec()) / 2
    corner_to_center = mid_tangent.perp()

    half_angle = line2.start_tangent() - polar_angle_from_vec(corner_to_center)

    distance = radius / half_angle.tan()

    new_end_1 = corner + line1.end_tangent().as_vec().scale(distance * ccw)
    new_start_2 = corner - line2.start_tangent().as_vec().scale(distance * ccw)

    return [
        LineSegment(line1.start, new_end_1),
        bulging_segment_from_start_tangent(new_end_1, new_start_2, line1.end_tangent()),
        LineSegment(new_start_2, line2.end),
    ]
