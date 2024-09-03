from polymorph_s2df import draw, XY_PLANE, sweep
from polymorph_num.ops import grid_gen_3d
import polymorph_num.expr as ir
import math
import time
import dataclasses
import collections

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
print(f"Build IR: {after - before:.2f}s")


expr_id = {}


def node_name(expr: ir.Expr) -> str:
    assert isinstance(expr, ir.Expr)
    if expr not in expr_id:
        val = len(expr_id)
        expr_id[expr] = str(val)
    return expr_id[expr]


def node_type(expr: ir.Expr) -> str:
    assert isinstance(expr, ir.Expr)
    match expr:
        case ir.Binary(_, _, op):
            return f"Binary{op.name}"
        case ir.Unary(_, op, _):
            return f"Unary{op.name}"
        case ir.Scalar(v):
            return f"Scalar {v}"
        case _:
            return type(expr).__name__


def topo(expr: ir.Expr) -> list[ir.Expr]:
    visited = set()
    result = []
    def visit(expr: ir.Expr):
        if expr in visited:
            return
        visited.add(expr)
        for edge in edges(expr):
            visit(edge)
        result.append(expr)
    visit(expr)
    return result

def edges(expr: ir.Expr) -> list[ir.Expr]:
    match expr:
        case ir.Binary(left, right, _):
            return [left, right]
        case ir.Unary(orig, _, _):
            return [orig]
        case ir.Broadcast(orig, _):
            return [orig]
        case ir.Sum(orig):
            return [orig]
        case ir.ComparisonIf(a, b, ctrue, cfalse, _):
            return [a, b, ctrue, cfalse]
        case ir.Param(_) | ir.Observation(_) | ir.Scalar(_) | ir.Arr(_):
            return []
        case (
            ir.GridX(_, _)
            | ir.GridY(_, _)
            | ir.GridX3d(_, _, _)
            | ir.GridY3d(_, _, _)
            | ir.GridZ3d(_, _, _)
        ):
            return []
        case _:
            raise ValueError(f"Unknown IR type: {type(expr)}")

from graphviz import Digraph
def draw_dot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})

    for v in topo(root):
        name = node_name(v)
        dot.node(name=name, label = "{ type %s }" % (node_type(v), ), shape='record')
        for to in edges(v):
            dot.edge(name, node_name(to))

    return dot


@dataclasses.dataclass
class Optimizer:
    cse: list[ir.Expr] = dataclasses.field(default_factory=list)

    def opt(self, expr: ir.Expr) -> ir.Expr:
        match expr:
            case ir.Param(_) | ir.Observation(_) | ir.Scalar(_) | ir.Arr(_):
                return False
            case ir.ComparisonIf(ir.Scalar(_), ir.Scalar(_), ctrue, cfalse, _):
                raise ValueError("ComparisonIf scalar")
            case ir.ComparisonIf(_):
                return False
            case (
                ir.GridX(_, _)
                | ir.GridY(_, _)
                | ir.GridX3d(_, _, _)
                | ir.GridY3d(_, _, _)
                | ir.GridZ3d(_, _, _)
            ):
                return False
            case (
                ir.Binary(ir.Scalar(0), x, ir.BinOp.Add)
                | ir.Binary(x, ir.Scalar(0), ir.BinOp.Add)
            ):
                expr.make_equal_to(x.find())
                return True
            case (
                ir.Binary(ir.Scalar(0), x, ir.BinOp.Mul)
                | ir.Binary(x, ir.Scalar(0), ir.BinOp.Mul)
            ):
                expr.make_equal_to(ir.Scalar(0))
                return True
            case (
                ir.Binary(ir.Scalar(1), x, ir.BinOp.Mul)
                | ir.Binary(x, ir.Scalar(1), ir.BinOp.Mul)
            ):
                expr.make_equal_to(x.find())
                return True
            case ir.Binary(ir.Scalar(l), ir.Scalar(r), ir.BinOp.Add):
                expr.make_equal_to(ir.Scalar(l + r))
                return True
            case ir.Binary(ir.Scalar(l), ir.Scalar(r), ir.BinOp.Mul):
                expr.make_equal_to(ir.Scalar(l * r))
                return True
            case ir.Binary(ir.Scalar(l), ir.Scalar(r), ir.BinOp.Div):
                expr.make_equal_to(ir.Scalar(l / r))
                return True
            case ir.Binary(ir.Scalar(l), ir.Scalar(r), ir.BinOp.Mod):
                expr.make_equal_to(ir.Scalar(l % r))
                return True
            case ir.Binary(ir.Scalar(l), ir.Scalar(r), ir.BinOp.Sub):
                expr.make_equal_to(ir.Scalar(l - r))
                return True
            case ir.Binary(ir.Scalar(l), ir.Scalar(r), ir.BinOp.ArcTan2):
                expr.make_equal_to(ir.Scalar(math.atan2(l, r)))
                return True
            case ir.Binary(left, right, op):
                if isinstance(left, ir.Scalar) and isinstance(right, ir.Scalar):
                    raise ValueError(f"Binary scalar: {left} {op} {right}")
                return False
            case ir.Unary(ir.Scalar(x), ir.UnOp.Sqrt, _):
                expr.make_equal_to(ir.Scalar(math.sqrt(x)))
                return True
            case ir.Unary(ir.Scalar(x), ir.UnOp.Cos, _):
                expr.make_equal_to(ir.Scalar(math.cos(x)))
                return True
            case ir.Unary(ir.Scalar(x), ir.UnOp.Sin, _):
                expr.make_equal_to(ir.Scalar(math.sin(x)))
                return True
            case ir.Unary(orig, op, consts):
                if isinstance(orig, ir.Scalar):
                    raise ValueError(f"Unary scalar: {op} {orig}")
                return False
            case ir.Broadcast(orig, dim):
                # TODO(max): This is slow
                # if isinstance(orig, ir.Scalar):
                #     expr.make_equal_to(ir.Arr([orig] * dim))
                #     return True
                return False
            case ir.Sum(orig):
                if isinstance(orig, ir.Scalar):
                    raise ValueError(f"Sum scalar: {orig}")
                if isinstance(orig, ir.Arr):
                    raise ValueError(f"Sum arr: {orig}")
                return False
            case _:
                raise ValueError(f"Unknown IR type: {type(expr)}")

    def spin_opt(self, expr: ir.Expr) -> ir.Expr:
        cycles = 0
        while True:
            changed = False
            for e in topo(expr.find()):
                changed |= self.opt(e.find())
            expr_opt = expr.find()
            if not changed:
                print(f"Optimization cycles: {cycles}")
                return expr_opt
            expr = expr_opt
            cycles += 1


kinds = collections.Counter()
for e in topo(expr):
    kinds[type(e)] += 1
print(kinds)

before = time.perf_counter()
optimizer = Optimizer()
expr_opt = optimizer.spin_opt(expr)
after = time.perf_counter()
print(f"Optimize IR: {after - before:.2f}s")

kinds = collections.Counter()
for e in topo(expr_opt):
    kinds[type(e)] += 1
print(kinds)
# print(optimizer.cse_hits)
# print(optimizer.kinds)

# print(draw_dot(expr_opt))
