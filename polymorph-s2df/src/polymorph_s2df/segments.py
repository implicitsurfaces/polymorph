from functools import cached_property

from polymorph_num import ops
from polymorph_num.angle import Angle, polar_angle
from polymorph_num.distance import Length
from polymorph_num.expr import Expr, Num
from polymorph_num.vec import Vec2


class Line:
    start: Vec2
    end: Vec2
    length: Length
    direction: Angle

    def __init__(self, start: Vec2, end: Vec2, length: Length, direction: Angle):
        self.start = start
        self.end = end
        self.length = length
        self.direction = direction

    @cached_property
    def entrance_angle(self):
        return self.direction

    @cached_property
    def exit_angle(self):
        return self.direction

    @cached_property
    def segment(self):
        return self.end - self.start

    def winding_number(self, p: Vec2) -> Expr:
        a = self.start - p
        b = self.end - p

        return ops.atan2(a.cross(b), a.dot(b))

    def distance(self, x: Num, y: Num) -> Expr:
        p = Vec2(x, y)
        start_to_p = p - self.start

        parametric_position = start_to_p.dot(self.segment) / self.segment.dot(
            self.segment
        )
        clamped_position = ops.clamp(parametric_position, 0, 1)

        projected_point = self.start + self.segment.scale(clamped_position)
        return (p - projected_point).norm()


def line(start: Vec2, end: Vec2) -> Line:
    segment = end - start
    return Line(start, end, Length(segment.norm()), polar_angle(segment.x, segment.y))


def line_polar(start: Vec2, angle: Angle, length: Length) -> Line:
    return Line(start, start + length.as_vec(angle), length, angle)


class Arc:
    start: Vec2
    end: Vec2
    start_tangent: Angle
    end_tangent: Angle
    center: Vec2
    radius: Length
    bulge: Expr
    chord: Line


def arc(line: Line, bulge: Expr):
    pass


def arc_center(line: Line, center: Vec2):
    pass


def arc_with_point(line: Line, third_point: Vec2):
    pass


def arc_start_tangent(line: Line, start_tangent: Angle):
    pass


def arc_end_tangent(line: Line, end_tangent: Angle):
    pass


def arc_tangents_center(
    center: Vec2, radius: Length, start_tangent: Angle, end_tangent: Angle
):
    pass


def arc_tangents_start(
    start_tangent: Angle, end_tangent: Angle, radius: Length, start: Vec2
):
    pass


def arc_tangents_end(
    start_tangent: Angle, end_tangent: Angle, radius: Length, end: Vec2
):
    pass

def arc_tangents_corner(
    start_tangent: Angle, end_tangent: Angle, radius: Length, corner: Vec2
):
    pass


type Segment = Line | Arc
