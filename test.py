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


kinds = collections.Counter()
for e in topo(expr):
    kinds[type(e)] += 1
print("Total nodes:", sum(kinds.values()), file=sys.stderr)
print(kinds, file=sys.stderr)

before = time.perf_counter()
optimizer = Optimizer()
expr = optimizer.spin_opt(expr)
print(f"Optimization cycles: {optimizer.cycles}", file=sys.stderr)
after = time.perf_counter()
print("Timers:", file=sys.stderr)
for measure, duration in sorted(
    optimizer.timers.items(), key=lambda x: x[1], reverse=True
):
    print(f"  {measure:15}: {duration:.2f}s", file=sys.stderr)
print(f"Optimize IR: {after - before:.2f}s", file=sys.stderr)

kinds = collections.Counter()
for e in topo(expr):
    kinds[type(e)] += 1
print("Total nodes:", sum(kinds.values()), file=sys.stderr)
print(kinds, file=sys.stderr)
print(draw_dot(expr))
before = time.perf_counter()
unit = Unit()
compiled_unit = unit.register("expr", expr).compile()
after = time.perf_counter()
print(f"JAX Compile: {after - before:.2f}s", file=sys.stderr)
