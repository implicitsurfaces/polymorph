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
    # .to_point((-1, 2 * space, 0), enable_rotation_mode)
    # .to_point((1, 3 * space, 0), enable_rotation_mode)
    # .to_point((-1, 4 * space, 0), enable_rotation_mode)
    .to_solid()
)

grid_x, grid_y, grid_z = grid_gen_3d(100, 100, 100)
expr = solid.distance(grid_x, grid_y, grid_z)
after = time.perf_counter()
# print(f"Build IR: {after - before:.2f}s")

# @dataclasses.dataclass(frozen=True)
# class Dist(ir.Expr):
#     dim: int
#     a: ir.Expr
#     b: ir.Expr


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
    cse_cache: dict[tuple[type, tuple[object, ...]], ir.Expr] = dataclasses.field(
        default_factory=dict
    )
    cse_hits: collections.Counter = dataclasses.field(
        default_factory=collections.Counter
    )
    visited: set[ir.Expr] = dataclasses.field(default_factory=set)
    kinds: collections.Counter = dataclasses.field(default_factory=collections.Counter)

    def construct(self, type_: type, *args) -> ir.Expr:
        key = (type_, args)
        result = self.cse_cache.get(key)
        if result is not None:
            self.cse_hits[type_] += 1
            return result
        result = type_(*args)
        self.cse_cache[key] = result
        # self.visited.add(result)
        return result

    def opt(self, expr: ir.Expr) -> ir.Expr:
        if expr in self.visited:
            return expr
        self.kinds[type(expr)] += 1
        self.visited.add(expr)
        match expr:
            case ir.Param(_) | ir.Observation(_) | ir.Scalar(_) | ir.Arr(_):
                return expr
            case (
                ir.GridX(_, _)
                | ir.GridY(_, _)
                | ir.GridX3d(_, _, _)
                | ir.GridY3d(_, _, _)
                | ir.GridZ3d(_, _, _)
            ):
                return expr
            case (
                ir.Binary(ir.Scalar(0), x, ir.BinOp.Add)
                | ir.Binary(x, ir.Scalar(0), ir.BinOp.Add)
            ):
                return self.opt(x)
            case (
                ir.Binary(ir.Scalar(0), x, ir.BinOp.Mul)
                | ir.Binary(x, ir.Scalar(0), ir.BinOp.Mul)
            ):
                return ir.Scalar(0)
            case (
                ir.Binary(ir.Scalar(1), x, ir.BinOp.Mul)
                | ir.Binary(x, ir.Scalar(1), ir.BinOp.Mul)
            ):
                return self.opt(x)
            case ir.Binary(ir.Scalar(l), ir.Scalar(r), ir.BinOp.Add):
                return self.construct(ir.Scalar, l + r)
            case ir.Binary(ir.Scalar(l), ir.Scalar(r), ir.BinOp.Mul):
                return self.construct(ir.Scalar, l * r)
            case ir.Binary(ir.Scalar(l), ir.Scalar(r), ir.BinOp.Sub):
                return self.construct(ir.Scalar, l - r)
            case ir.Binary(left, right, op):
                if isinstance(left, ir.Scalar) and isinstance(right, ir.Scalar):
                    raise ValueError(f"Binary scalar: {left} {op} {right}")
                return self.construct(ir.Binary, self.opt(left), self.opt(right), op)
            case ir.Unary(ir.Scalar(x), ir.UnOp.Sqrt, _):
                return self.construct(ir.Scalar, math.sqrt(x))
            case ir.Unary(ir.Scalar(x), ir.UnOp.Cos, _):
                return self.construct(ir.Scalar, math.cos(x))
            case ir.Unary(ir.Scalar(x), ir.UnOp.Sin, _):
                return self.construct(ir.Scalar, math.sin(x))
            case ir.Unary(ir.Binary(ir.Binary(a, b, ir.BinOp.Mul), ir.Binary(c,
                                                                             d,
                                                                             ir.BinOp.Mul), ir.BinOp.Add) as orig, ir.UnOp.Sqrt as op, consts):
                # print("Sqrt of add of mul", type(a), type(b), type(c), type(d))
                # if a is b and c is d:
                #     assert a.dim == c.dim
                #     return self.construct(Dist, a.dim, self.opt(a), self.opt(b))
                return self.construct(ir.Unary, self.opt(orig), op, consts)
            case ir.Unary(orig, op, consts):
                if isinstance(orig, ir.Scalar):
                    raise ValueError(f"Unary scalar: {op} {orig}")
                return self.construct(ir.Unary, self.opt(orig), op, consts)
            case ir.Broadcast(orig, dim):
                if isinstance(orig, ir.Scalar):
                    return ir.Arr([orig] * dim)
                return self.construct(ir.Broadcast, self.opt(orig), dim)
            case ir.Sum(orig):
                if isinstance(orig, ir.Scalar):
                    raise ValueError(f"Sum scalar: {orig}")
                if isinstance(orig, ir.Arr):
                    raise ValueError(f"Sum arr: {orig}")
                return self.construct(ir.Sum, self.opt(orig))
            case ir.Random(dim, low, high):
                return self.construct(ir.Random, dim, self.opt(low), self.opt(high))
            case ir.ComparisonIf(a, b, ctrue, cfalse, op):
                if isinstance(a, ir.Scalar) and isinstance(b, ir.Scalar):
                    raise ValueError(f"Comparison scalar: {a} {op} {b}")
                return self.construct(
                    ir.ComparisonIf,
                    self.opt(a),
                    self.opt(b),
                    self.opt(ctrue),
                    self.opt(cfalse),
                    op,
                )
            # case Dist(dim, a, b):
            #     return self.construct(Dist, dim, self.opt(a), self.opt(b))
            case _:
                raise ValueError(f"Unknown IR type: {type(expr)}")

    def spin_opt(self, expr: ir.Expr) -> ir.Expr:
        cycles = 0
        while True:
            expr_opt = self.opt(expr)
            if expr_opt == expr:
                # print(f"Optimization cycles: {cycles}")
                return expr_opt
            expr = expr_opt
            cycles += 1


optimizer = Optimizer()
expr_opt = optimizer.spin_opt(expr)
# print(optimizer.cse_hits)
# print(optimizer.kinds)

print(draw_dot(expr_opt))
