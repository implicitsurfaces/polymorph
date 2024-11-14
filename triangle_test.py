from polymorph_num import ops
from polymorph_num import vec
from polymorph_num import expr
from polymorph_num import unit
from polymorph_num import optimizer
import math

side = ops.observation("side")

origin = vec.as_vec2((0,0))
deg90 = vec.as_vec2((0,1))
deg45 = vec.as_vec2((math.sqrt(2)/2,math.sqrt(2)/2))

def angle_add(a: vec.Vec2, b: vec.Vec2) -> vec.Vec2:
    return vec.as_vec2((
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x))
    
def right_triangle_distance(heading: vec.Vec2) -> expr.Expr:
    pt1 = origin + heading.scale(side*3)
    pt2 = pt1 + angle_add(heading, deg90).scale(side*4)
    dist1 = (pt1 - origin).norm()
    dist2 = (pt2 - pt1).norm()
    dist3 = (origin - pt2).norm()
    return dist1 + dist2 + dist3

def compute(e: expr.Expr):
    return unit.Unit(["side"]).register("e", e).compile().observe({"side": 1.0}).evaluate("e")

e90 = right_triangle_distance(deg90)
e45 = right_triangle_distance(deg45)

e90_opt = optimizer.Optimizer().spin_opt(e90)
print(compute(e90))
print(compute(e90_opt))
print(e90_opt)
