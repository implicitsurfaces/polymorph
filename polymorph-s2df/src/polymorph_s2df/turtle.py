from polymorph_num import ops as ops
from polymorph_num.expr import ZERO, Num, as_expr
from polymorph_num.expr import Expr as Expr
from polymorph_num.vec import Vec2, as_vec2
from .paths import (
    ClosedPath,
    LineSegment,
    PathSegment,
)

class Turtle:
    position: Vec2
    heading: Expr
    segments: list[PathSegment]
    loss: Expr

    def __init__(self):
        self.position = Vec2(0.,0.)
        self.initial_position = self.position
        self.heading = as_expr(0.)
        self.segments = []
        self.loss = as_expr(0.)
    
    def forward(self, distance: Num):
        old_position = self.position
        delta = Vec2(distance, 0).rotate(self.heading)
        self.position = self.position + delta
        self.segments.append(LineSegment(old_position, self.position))
        return self.position

    def turnLeft(self, radians: Num):
        self.heading -= radians

    def turnRight(self, radians: Num):
        self.heading += radians       

    def assert_closed(self):
        delta = self.position - self.initial_position
        dist_sq = delta.norm_squared()
        self.loss += dist_sq
        return ClosedPath(self.segments) 
    
    def assert_distance(self, pt1, pt2, distance):
        delta = pt1 - pt2
        d = delta.norm()
        error = distance - d
        self.loss += (error*error)

    def close(self):
        self.segments.append(LineSegment(self.position, self.initial_position))
        return ClosedPath(self.segments) 