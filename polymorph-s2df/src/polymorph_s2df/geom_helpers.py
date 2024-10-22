from polymorph_num.angle import Angle, angle_from_cos, polar_angle_from_vec
from polymorph_num.expr import Expr
from polymorph_num.vec import Vec2

from polymorph_s2df.paths import BulgingSegment, LineSegment, PathSegment


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


def quadratic_biarc(x0: Expr, y0: Expr, x1: Expr, y1: Expr, x2: Expr, y2: Expr):
    p0 = Vec2(x0, y0)
    p1 = Vec2(x1, y1)
    p2 = Vec2(x2, y2)

    w0 = (p2 - p1).norm()
    w1 = (p2 - p0).norm()
    w2 = (p1 - p0).norm()

    perimeter = w0 + w1 + w2

    c = (p0.scale(w0) + p1.scale(w1) + p2.scale(w2)) / perimeter

    theta0 = polar_angle_from_vec(p2 - p0)
    theta1 = polar_angle_from_vec(p2 - p1).opposite()

    return [
        bulging_segment_from_start_tangent(p0, c, theta0),
        bulging_segment_from_end_tangent(c, p2, theta1),
    ]


def cubic_biarc(
    x0: Expr, y0: Expr, x1: Expr, y1: Expr, x2: Expr, y2: Expr, x3: Expr, y3: Expr
):
    p0 = Vec2(x0, y0)
    p1 = Vec2(x1, y1)
    p2 = Vec2(x2, y2)
    p3 = Vec2(x3, y3)

    tgt_start = p2 - p0
    tgt_end = p1 - p3

    theta0 = polar_angle_from_vec(tgt_start)
    theta1 = polar_angle_from_vec(tgt_end)

    locus = _biarc_locus_center(p0, theta0, p1, theta1)

    position_ratio = tgt_start.norm() / (tgt_start.norm() + tgt_end.norm())

    chord = p1 - p0
    postion_on_chord = p0 + chord.scale(position_ratio)

    c_angle = polar_angle_from_vec(postion_on_chord - locus)
    c = locus + c_angle.as_vec().scale((locus - p0).norm())

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
    ccw = (line2.start_tangent() - line1.end_tangent()).sin().sign()

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


def project_point_on_line(point: Vec2, p: Vec2, v: Vec2):
    return p + v.scale((point - p).dot(v))


def fillet_line_arc(
    line: LineSegment, arc: BulgingSegment, radius: Expr
) -> list[PathSegment]:
    corner = line.end
    ccw = (arc.start_tangent() - line.end_tangent()).sin().sign()
    arc_orientation = arc.bulge.sign()

    center = arc.center

    # we define the parallel line to the segment of line. Its direction is the
    # same as the segment, but its point is shifted
    line_direction = line.end_tangent().as_vec()
    parallel_p = corner + line_direction.perp().scale(ccw * radius)

    # we project the center of the arc on the parallel line, to create a triangle
    projected_center = project_point_on_line(center, parallel_p, line_direction)

    side = (center - projected_center).norm()
    hypothenuse = arc.radius - (ccw * arc_orientation * radius)
    last_side = (hypothenuse * hypothenuse - side * side).sqrt()

    fillet_center = projected_center + line_direction.scale(
        arc_orientation * last_side * ccw
    )

    line_end = fillet_center - line_direction.perp().scale(radius * ccw)

    fillet_center_to_arc_center = center - fillet_center
    fillet_center_to_arc_center = (
        fillet_center_to_arc_center / fillet_center_to_arc_center.norm()
    )
    arc_start = fillet_center - fillet_center_to_arc_center.scale(
        arc_orientation * radius * ccw
    )

    fillet_arc = bulging_segment_from_start_tangent(
        line_end, arc_start, line.end_tangent()
    )

    return [
        LineSegment(line.start, line_end),
        fillet_arc,
        bulging_segment_from_start_tangent(
            arc_start, arc.end, fillet_arc.end_tangent()
        ),
    ]


def fillet_arc_line(
    arc: BulgingSegment, line: LineSegment, radius: Expr
) -> list[PathSegment]:
    fillet = fillet_line_arc(line.reversed(), arc.reversed(), radius)
    fillet.reverse()
    return [s.reversed() for s in fillet]


def fillet_arc_arc(
    arc1: BulgingSegment, arc2: BulgingSegment, radius: Expr
) -> list[PathSegment]:
    ccw = (arc2.start_tangent() - arc1.end_tangent()).sin().sign()
    arc1_orientation = arc1.bulge.sign()
    arc2_orientation = arc2.bulge.sign()

    centers_distance = (arc2.center - arc1.center).norm()
    centers_direction = (arc2.center - arc1.center) / centers_distance

    side_1 = arc1.radius - (arc1_orientation * radius * ccw)
    side_2 = arc2.radius - (arc2_orientation * radius * ccw)

    side_1_cos = (
        side_1 * side_1 + centers_distance * centers_distance - side_2 * side_2
    ) / (2 * centers_distance * side_1)

    angle_1 = angle_from_cos(side_1_cos)

    projected_fillet_center = arc1.center + centers_direction.scale(
        angle_1.cos() * side_1
    )

    fillet_center = projected_fillet_center + centers_direction.perp().scale(
        angle_1.sin() * side_1 * arc1_orientation * arc2_orientation * ccw
    )

    r1_direction = fillet_center - arc1.center
    r1_direction = r1_direction / r1_direction.norm()
    arc_1_end = fillet_center + r1_direction.scale(radius * ccw * arc1_orientation)

    r2_direction = fillet_center - arc2.center
    r2_direction = r2_direction / r2_direction.norm()
    arc_2_start = fillet_center + r2_direction.scale(radius * ccw * arc2_orientation)

    first_arc = bulging_segment_from_start_tangent(
        arc1.start, arc_1_end, arc1.start_tangent()
    )
    fillet_arc = bulging_segment_from_start_tangent(
        arc_1_end, arc_2_start, first_arc.end_tangent()
    )
    second_arc = bulging_segment_from_start_tangent(
        arc_2_start, arc2.end, fillet_arc.end_tangent()
    )

    return [
        first_arc,
        fillet_arc,
        second_arc,
    ]
