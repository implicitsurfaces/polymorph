from polymorph_s2df import draw, XY_PLANE, sweep
from polymorph_num.ops import grid_gen_3d
import polymorph_num.expr as ir
import math
import time
import dataclasses
import collections
import sys

before = time.perf_counter()
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
#     sweep(profile, plane)
#     .to_point((1, space, 0), enable_rotation_mode)
#     # .to_point((-1, 2 * space, 0), enable_rotation_mode)
#     # .to_point((1, 3 * space, 0), enable_rotation_mode)
#     # .to_point((-1, 4 * space, 0), enable_rotation_mode)
#     .to_solid()
# )
# 
# grid_x, grid_y, grid_z = grid_gen_3d(100, 100, 100)
# expr = solid.distance(grid_x, grid_y, grid_z)
side_length = ir.Param(0)
corners = [
    [ir.Scalar(0), ir.Scalar(0)],
    [ir.Scalar(0), side_length],
    [side_length, side_length],
    [side_length, ir.Scalar(0)],
]
def distance(p0, p1):
    return ir.Unary(
            ir.Binary(
                ir.Binary(
                    ir.Binary(p0[0], p1[0], ir.BinOp.Sub),
                    ir.Binary(p0[0], p1[0], ir.BinOp.Sub),
                    ir.BinOp.Mul),
                ir.Binary(
                    ir.Binary(p0[1], p1[1], ir.BinOp.Sub),
                    ir.Binary(p0[1], p1[1], ir.BinOp.Sub),
                    ir.BinOp.Mul),
                ir.BinOp.Add),
        ir.UnOp.Sqrt,
        (),
    )
expr = distance(corners[0], corners[1]) + \
        distance(corners[1], corners[2]) + \
        distance(corners[2], corners[3]) + \
        distance(corners[3], corners[0]) + \
        distance(corners[0], corners[2]) + \
        distance(corners[1], corners[3])

after = time.perf_counter()
print(f"Build IR: {after - before:.2f}s", file=sys.stderr)


def node_name(expr: ir.Expr) -> str:
    return f"v{expr.id}"


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
        if expr.id in visited:
            return
        visited.add(expr.id)
        for edge in edges(expr):
            visit(edge.find())
        result.append(expr)
    visit(expr.find())
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
        dot.node(name=name, label = "{ type %s | range [%s,%s] }" % (node_type(v), v.range[0], v.range[1]), shape='record')
        for to in edges(v):
            dot.edge(name, node_name(to))

    return dot


@dataclasses.dataclass
class Optimizer:
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
                ir.Binary(x, ir.Scalar(0), ir.BinOp.Sub)
            ):
                expr.make_equal_to(x.find())
                return True
            case ir.Binary(x, y, ir.BinOp.Sub) if x is y:
                expr.make_equal_to(ir.Scalar(0))
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
            case ir.Binary(l, r, ir.BinOp.Mul) if l is r:
                expr.make_equal_to(ir.Unary(l, ir.UnOp.Sqr))
                return True
            case ir.Binary(ir.Scalar(_), ir.Scalar(_), op):
                raise ValueError(f"Binary scalar: {left} {op} {right}")
            case ir.Binary(ir.Arr(_), ir.Arr(_), op):
                raise ValueError(f"Binary arr: {left} {op} {right}")
            case ir.Binary(_, _, _):
                return False
            case ir.Unary(ir.Unary(x, ir.UnOp.Sqr, _), ir.UnOp.Sqrt, _):
                expr.make_equal_to(ir.Unary(x, ir.UnOp.Abs, ()))
                return True
            case ir.Unary(ir.Scalar(x), ir.UnOp.Sqrt, _):
                expr.make_equal_to(ir.Scalar(math.sqrt(x)))
                return True
            case ir.Unary(ir.Scalar(x), ir.UnOp.Cos, _):
                expr.make_equal_to(ir.Scalar(math.cos(x)))
                return True
            case ir.Unary(ir.Scalar(x), ir.UnOp.Sin, _):
                expr.make_equal_to(ir.Scalar(math.sin(x)))
                return True
            case ir.Unary(ir.Scalar(x), ir.UnOp.Sqr, _):
                expr.make_equal_to(ir.Scalar(x * x))
                return True
            # case ir.Unary(ir.Unary(x, ir.UnOp.Sqr, _), ir.UnOp.Sqrt, _):
            #     raise ValueError(f"Unary unary sqr sqrt: {x}")
            # case ir.Unary(
            #         ir.Binary(
            #             ir.Unary(l, ir.UnOp.Sqr, _),
            #             ir.Unary(r, ir.UnOp.Sqr, _),
            #             ir.BinOp.Add),
            #         ir.UnOp.Sqrt):
            #     expr.make_equal_to(ir.Binary(l, r, ir.BinOp.RMS))
            #     return True
            case ir.Unary(ir.Scalar(_), op, consts):
                raise ValueError(f"Unary scalar: {op} {orig}")
            case ir.Unary(orig, op, consts):
                return False
            case ir.Broadcast(ir.Scalar(x), dim):
                return False
            case ir.Broadcast(orig, dim):
                # # TODO(max): This is slow
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
                absint_range_one(e.find())
            expr_opt = expr.find()
            if not changed:
                print(f"Optimization cycles: {cycles}", file=sys.stderr)
                return expr_opt
            expr = expr_opt
            cycles += 1


def absint_sqrt(x: float) -> float:
    if x < 0:
        return -math.inf
    return math.sqrt(x)


def absint_range_one(expr: ir.Expr) -> None:
    match expr:
        case ir.Scalar(v):
            expr.update_range(v, v)
        case ir.Param(_):
            expr.update_range(-math.inf, math.inf)
        case ir.Binary(left, right, ir.BinOp.Add):
            left_min, left_max = left.range
            right_min, right_max = right.range
            expr.update_range(left_min + right_min, left_max + right_max)
        case ir.Binary(left, right, ir.BinOp.Mul) if left == right:
            left_min, left_max = left.range
            expr.update_range(0, left_max * left_max)
        case ir.Binary(left, right, ir.BinOp.Mul):
            left_min, left_max = left.range
            right_min, right_max = right.range
            expr.update_range(
                    min(left_min * right_min,
                        left_min * right_max,
                        left_max * right_min,
                        left_max * right_max),
                    max(left_min * right_min,
                        left_min * right_max,
                        left_max * right_min,
                        left_max * right_max))
        case ir.Binary(left, right, ir.BinOp.Sub):
            left_min, left_max = left.range
            right_min, right_max = right.range
            expr.update_range(left_min - right_max, left_max - right_min)
        case ir.Unary(orig, ir.UnOp.Sqrt, _):
            orig_min, orig_max = orig.range
            expr.update_range(absint_sqrt(orig_min), absint_sqrt(orig_max))
        case ir.Binary(left, right, op):
            raise ValueError(f"Binary: {op}")
        case _:
            raise ValueError(f"Unknown IR type: {type(expr)}")


def absint_range(expr: ir.Expr) -> None:
    for e in topo(expr):
        absint_range_one(e)


kinds = collections.Counter()
for e in topo(expr):
    kinds[type(e)] += 1
print(kinds, file=sys.stderr)

before = time.perf_counter()
optimizer = Optimizer()
expr_opt = optimizer.spin_opt(expr)
after = time.perf_counter()
print(f"Optimize IR: {after - before:.2f}s", file=sys.stderr)

kinds = collections.Counter()
for e in topo(expr_opt):
    kinds[type(e)] += 1
print(kinds, file=sys.stderr)
print(draw_dot(expr_opt))
