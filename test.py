from __future__ import annotations
from polymorph_s2df import draw, XY_PLANE, sweep
from polymorph_num.ops import grid_gen_3d
from polymorph_num.unit import Unit
from polymorph_num.optimizer import Optimizer, topo, edges
import polymorph_num.expr as ir
import math
import time
import dataclasses
import collections
import sys

before = time.perf_counter()
profile = (
    draw((-0.1, 0))
    .horizontal_line(0.2)
    .line_to((0.05, 0.3))
    .close()
    .rotate(math.pi / 2)
)

plane = XY_PLANE.translateTo((-1, 0, 0))
enable_rotation_mode = True
space = 0.2
solid = (
    sweep(profile, plane)
    .to_point((1, space, 0), enable_rotation_mode)
    .to_point((-1, 2 * space, 0), enable_rotation_mode)
    .to_point((1, 3 * space, 0), enable_rotation_mode)
    .to_point((-1, 4 * space, 0), enable_rotation_mode)
    .to_solid()
)

grid_x, grid_y, grid_z = grid_gen_3d(100, 100, 100)
expr = solid.distance(grid_x, grid_y, grid_z)
after = time.perf_counter()
print(f"Build IR: {after - before:.2f}s", file=sys.stderr)


def node_name(expr: ir.Expr) -> str:
    return f"v{expr.id}"


def node_type(expr: ir.Expr) -> str:
    assert isinstance(expr, ir.Expr)
    match expr:
        case ir.Binary(_, _, op):
            return f"Binary{op.name}[{expr.dim}]"
        case ir.Unary(_, op, _):
            return f"Unary{op.name}[{expr.dim}]"
        case ir.Scalar(v):
            return f"Scalar {v}"
        case ir.ComparisonIf(a, b, _, _, op):
            return f"ComparisonIf[{expr.dim}] {a.range} {op.name} {b.range}"
        case _:
            return f"{type(expr).__name__}[{expr.dim}]"


from graphviz import Digraph


def draw_dot(root, format="svg", rankdir="LR"):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ["LR", "TB"]
    dot = Digraph(format=format, graph_attr={"rankdir": rankdir})

    for v in topo(root):
        name = node_name(v)
        dot.node(
            name=name,
            label="{ type %s | range [%s,%s] }"
            % (node_type(v), v.range[0], v.range[1]),
            shape="record",
        )
        for to in edges(v):
            dot.edge(name, node_name(to))

    return dot


# kinds = collections.Counter()
# for e in topo(expr):
#     kinds[type(e)] += 1
# print("Total nodes:", sum(kinds.values()), file=sys.stderr)
# print(kinds, file=sys.stderr)
# 
# before = time.perf_counter()
# optimizer = Optimizer()
# expr = optimizer.spin_opt(expr)
# print(f"Optimization cycles: {optimizer.cycles}", file=sys.stderr)
# after = time.perf_counter()
# print("Timers:", file=sys.stderr)
# for measure, duration in sorted(
#     optimizer.timers.items(), key=lambda x: x[1], reverse=True
# ):
#     print(f"  {measure:15}: {duration:.2f}s", file=sys.stderr)
# print(f"Optimize IR: {after - before:.2f}s", file=sys.stderr)
# 
# kinds = collections.Counter()
# for e in topo(expr):
#     kinds[type(e)] += 1
# print("Total nodes:", sum(kinds.values()), file=sys.stderr)
# print(kinds, file=sys.stderr)
# print(draw_dot(expr))
# before = time.perf_counter()
# unit = Unit()
# compiled_unit = unit.register("expr", expr).compile()
# after = time.perf_counter()
# print(f"JAX Compile: {after - before:.2f}s", file=sys.stderr)

# import polyscope as ps
# import interactive_polyscope
# import polymorph_s2df as sdf
# from polymorph_s2df.devutils import *
# from polymorph_s2df import *
# from polymorph_num.expr import to_str
# import math
# 
# ps.init()
# 
# profile = (
#     draw((-0.1, 0))
#     .horizontal_line(0.2)
#     .line_to((0.05, 0.3))
#     .close()
#     .rotate(math.pi / 2)
# )
# 
# plane = XY_PLANE.translateTo((-1, 0, 0))
# enable_rotation_mode = True
# space = 0.2
# solid = (
# 	sweep(profile, plane)
#     .to_point((1, space, 0), enable_rotation_mode)
#     .to_point((-1, 2*space, 0), enable_rotation_mode)
#     .to_point((1, 3*space, 0), enable_rotation_mode)
#     .to_point((-1, 4*space, 0), enable_rotation_mode)
#     .to_solid()
# )
# 
# print("Rendering", file=sys.stderr)
# before = time.perf_counter()
# render_solid(
#     solid, 
#     bounds=(-1.5, 2), 
#     n=100, 
#     show_distances=False, 
#     eval_fn=eval_expr)
# after = time.perf_counter()
# print(f"Rendered ({after - before:.2f}s)", file=sys.stderr)
# ps.show()

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
    return unit.Unit(["side"]).register("e", e).compile().observe({"side": 2.0}).evaluate("e")

e90 = right_triangle_distance(deg90)
e45 = right_triangle_distance(deg45)

@dataclasses.dataclass
class Monomial:
    var: "Polynomial" | ir.Expr | None
    power: int
    coefficient: float

    def negate(self) -> Monomial:
        return Monomial(self.var, self.power, -self.coefficient)

    def __hash__(self) -> int:
        return hash((self.var, self.power, self.coefficient))

    def normalize(self) -> Monomial:
        if isinstance(self.var, Polynomial):
            return Monomial(self.var.normalize(), self.power, self.coefficient)
        return self

@dataclasses.dataclass
class Polynomial:
    components: tuple[Monomial]

    def __hash__(self) -> int:
        return hash(self.components)

    def add(self, other: Polynomial) -> Polynomial:
        return Polynomial(self.components + other.components).normalize()

    def sub(self, other: Polynomial) -> Polynomial:
        return Polynomial(self.components + tuple([c.negate() for c in
                                             other.components])).normalize()

    def mul(self, other: Polynomial) -> Polynomial:
        result = []
        for m0 in self.components:
            for m1 in other.components:
                result.append(Monomial(
                    var=m0.var,
                    power=m0.power + m1.power,
                    coefficient=m0.coefficient * m1.coefficient,
                ))
                result.append(Monomial(
                    var=m1.var,
                    power=m0.power + m1.power,
                    coefficient=m0.coefficient * m1.coefficient,
                ))
        return Polynomial(tuple(result)).normalize()

    def normalize(self) -> Polynomial:
        self.components = tuple(c.normalize() for c in self.components)
        unique = {}
        for c in self.components:
            if c.coefficient == 0:
                continue
            key = (c.var, c.power)
            if key in unique:
                unique[key] += c.coefficient
            else:
                unique[key] = c.coefficient
        return Polynomial(tuple([
            Monomial(var=var, power=power, coefficient=coeff)
            for (var, power), coeff in unique.items()
        ]))


def to_poly(expr: ir.Expr) -> Polynomial:
    if isinstance(expr, ir.Scalar):
        return Polynomial((Monomial(None, 0, expr.value),))
    if isinstance(expr, ir.Observation):
        return Polynomial((Monomial(expr, 1, 1),))
    if isinstance(expr, ir.Binary):
        left = to_poly(expr.left)
        right = to_poly(expr.right)
        if expr.op is ir.BinOp.Add:
            return left.add(right)
        if expr.op is ir.BinOp.Sub:
            return left.sub(right)
        if expr.op is ir.BinOp.Mul:
            return left.mul(right)
        raise ValueError(f"Unsupported binary expression: {expr.op}")
    if isinstance(expr, ir.Unary):
        orig = to_poly(expr.orig)
        if expr.op is ir.UnOp.Sqrt:
            return Polynomial((Monomial(orig, 0.5, 1),))
        raise ValueError(f"Unsupported unary expression: {expr.op}")
    raise ValueError(f"Unsupported expression: {type(expr)}")

# print(draw_dot(e90))
print("Unoptimized:", compute(e90), file=sys.stderr)
# print("Unoptimized:", compute(e45), file=sys.stderr)
e90_opt = optimizer.Optimizer().spin_opt(e90)
print("Optimized:", compute(e90_opt), file=sys.stderr)
# e45_opt = optimizer.Optimizer().spin_opt(e45)
# print("Optimized:", compute(e45_opt), file=sys.stderr)
print(to_poly(e90_opt))
print(draw_dot(e90_opt.find()))
