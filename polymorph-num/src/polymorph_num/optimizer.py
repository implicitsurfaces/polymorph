import polymorph_num.expr as ir
import dataclasses
import collections
import time
import math


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


def const(val, dim):
    assert isinstance(val, (int, float))
    scalar = ir.Scalar(val)
    if dim == 1:
        return scalar
    return ir.Broadcast(scalar, dim)


class Timer:
    def __init__(self, opt, name):
        self.name = name
        self.opt = opt

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.opt.timers[self.name] += self.end - self.start


@dataclasses.dataclass
class Optimizer:
    timers: dict[str, float] = dataclasses.field(default_factory=collections.Counter)
    cycles: int = 0

    def opt(self, expr: ir.Expr) -> ir.Expr:
        match expr:
            case ir.Param(_) | ir.Observation(_) | ir.Scalar(_) | ir.Arr(_):
                return False
            case _ if expr.range[0] == expr.range[1] and not isinstance(
                expr, (ir.Scalar, ir.Broadcast)
            ):
                expr.make_equal_to(const(expr.range[0], expr.dim))
                return True
            case (
                ir.Binary(ir.Scalar(0), x, ir.BinOp.Add)
                | ir.Binary(x, ir.Scalar(0), ir.BinOp.Add)
            ):
                expr.make_equal_to(x.find())
                return True
            case (
                ir.Binary(ir.Broadcast(ir.Scalar(0), dim), x, ir.BinOp.Add)
                | ir.Binary(x, ir.Broadcast(ir.Scalar(0), dim), ir.BinOp.Add)
            ):
                assert x.dim == dim
                expr.make_equal_to(x.find())
                return True
            case ir.Binary(x, ir.Scalar(0), ir.BinOp.Sub):
                expr.make_equal_to(x.find())
                return True
            case ir.Binary(x, y, ir.BinOp.Sub) if x is y:
                expr.make_equal_to(const(0, expr.dim))
                return True
            case (
                ir.Binary(ir.Scalar(0), x, ir.BinOp.Mul)
                | ir.Binary(x, ir.Scalar(0), ir.BinOp.Mul)
            ):
                expr.make_equal_to(const(0, expr.dim))
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
            case ir.Binary(
                ir.Broadcast(ir.Scalar(left), left_dim),
                ir.Broadcast(ir.Scalar(right), right_dim),
                op,
            ):
                assert left_dim == right_dim
                expr.make_equal_to(
                    ir.Broadcast(
                        ir.Binary(ir.Scalar(left), ir.Scalar(right), op), left_dim
                    )
                )
                return True
            case ir.Binary(left, right, ir.BinOp.Min):
                left_min, left_max = left.range
                right_min, right_max = right.range
                if left_max < right_min:
                    expr.make_equal_to(left)
                    return True
                if right_max < left_min:
                    expr.make_equal_to(right)
                    return True
                return False
            case ir.Binary(left, right, ir.BinOp.Max):
                left_min, left_max = left.range
                right_min, right_max = right.range
                if left_min > right_max:
                    expr.make_equal_to(left)
                    return True
                if right_min > left_max:
                    expr.make_equal_to(right)
                    return True
                return False
            case ir.Unary(ir.Broadcast(ir.Scalar(_) as x, x_dim), op, consts):
                expr.make_equal_to(ir.Broadcast(ir.Unary(x, op, consts), x_dim))
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
            case ir.ComparisonIf(a, b, ctrue, cfalse, ir.ComparisonOp.Gt):
                a_min, a_max = a.range
                b_min, b_max = b.range
                if a_min > b_max:
                    expr.make_equal_to(ctrue)
                    return True
                if a_max <= b_min:
                    expr.make_equal_to(cfalse)
                    return True
                return False
            case ir.ComparisonIf(a, b, ctrue, cfalse, ir.ComparisonOp.Ge):
                a_min, a_max = a.range
                b_min, b_max = b.range
                if a_min >= b_max:
                    expr.make_equal_to(ctrue)
                    return True
                if a_max < b_min:
                    expr.make_equal_to(cfalse)
                    return True
                return False
            case ir.ComparisonIf(a, b, ctrue, cfalse, ir.ComparisonOp.Eq):
                a_min, a_max = a.range
                b_min, b_max = b.range
                if a_min == b_max and a_max == b_min:
                    expr.make_equal_to(ctrue)
                    return True
                if a_max < b_min or a_min > b_max:
                    expr.make_equal_to(cfalse)
                    return True
                return False
            case _:
                return False

    def spin_opt(self, expr: ir.Expr) -> ir.Expr:
        print("Before:", len(topo(expr.find())))
        cycles = 0
        while True:
            changed = False
            with self.timer("topo"):
                exprs = topo(expr.find())
            for e in exprs:
                dim_before = e.dim
                range_before = e.range
                with self.timer("opt"):
                    changed |= self.opt(e.find())
                dim_after = e.find().dim
                assert (
                    dim_before == dim_after
                ), f"dim changed in optimization; was {dim_before}, now {dim_after}"
                with self.timer("absint_range"):
                    changed |= absint_range_one(e.find())
                range_after = e.find().range
                assert (
                    range_before[0] <= range_after[0]
                ), f"range min decreased in optimization; was {range_before[0]}, now {range_after[0]}"
                assert (
                    range_before[1] >= range_after[1]
                ), f"range max increased in optimization; was {range_before[1]}, now {range_after[1]}"
            # with self.timer("cse"):
            #     changed |= cse(expr)
            expr_opt = expr.find()
            if not changed:
                self.cycles = cycles
                print("After:", len(topo(expr.find())))
                return expr_opt
            expr = expr_opt
            cycles += 1

    def timer(self, name):
        return Timer(self, name)


def absint_sqrt(x: float) -> float:
    if x < 0:
        raise ValueError(f"sqrt of negative number: {x}")
        return -math.inf
    return math.sqrt(x)


def absint_mul(x: float, y: float) -> float:
    if x == 0 or y == 0:
        # Handle 0*inf, 0*-inf
        return 0
    return x * y


def absint_add(lr: tuple[float, float], rr: tuple[float, float]) -> tuple[float, float]:
    lmin, lmax = lr
    rmin, rmax = rr
    if (lmin == math.inf and rmin == -math.inf) or (
        lmin == -math.inf and rmin == math.inf
    ):
        return -math.inf, math.inf
    return lmin + rmin, lmax + rmax


def absint_sub(lr: tuple[float, float], rr: tuple[float, float]) -> tuple[float, float]:
    lmin, lmax = lr
    rmin, rmax = rr
    if (lmin == math.inf and rmax == math.inf) or (
        lmin == -math.inf and rmax == -math.inf
    ):
        return -math.inf, math.inf
    return lmin - rmax, lmax - rmin


def absint_abs(x: float) -> float:
    if x == -math.inf:
        return 0
    return abs(x)


def absint_range_one(expr: ir.Expr) -> None:
    match expr:
        case ir.Scalar(v):
            return expr.update_range(v, v)
        case ir.GridX(width, height) | ir.GridY(width, height):
            return expr.update_range(0, max(width, height))
        case (
            ir.GridX3d(width, height, depth)
            | ir.GridY3d(width, height, depth)
            | ir.GridZ3d(width, height, depth)
        ):
            return expr.update_range(0, max(width, height, depth))
        case ir.Binary(left, right, ir.BinOp.Add):
            return expr.update_range(*absint_add(left.range, right.range))
        case ir.Binary(left, right, ir.BinOp.Mul):
            left_min, left_max = left.range
            right_min, right_max = right.range
            values = [
                absint_mul(left_min, right_min),
                absint_mul(left_min, right_max),
                absint_mul(left_max, right_min),
                absint_mul(left_max, right_max),
            ]
            new_range = min(values), max(values)
            if left is right:
                return expr.update_range(0, max(values))
            return expr.update_range(*new_range)
        case ir.Binary(left, right, ir.BinOp.Sub):
            return expr.update_range(*absint_sub(left.range, right.range))
        case ir.Binary(left, right, ir.BinOp.ArcTan2):
            # TODO(max): Improve range if left/right ranges known
            return expr.update_range(-math.pi, math.pi)
        # case ir.Binary(left, right, ir.BinOp.Mod):
        #     left_min, left_max = left.range
        #     right_min, right_max = right.range
        #     # TODO(max): Is this necessarily positive? probably not
        #     return expr.update_range(0, right_max)
        # case ir.Binary(left, right, ir.BinOp.Div):
        #     left_min, left_max = left.range
        #     right_min, right_max = right.range
        #     if right_min <= 0 <= right_max:
        #         return False
        #     if any(math.isinf(x) for x in (left_min, left_max, right_min, right_max)):
        #         return False
        #     values = [left_min / right_min, left_min / right_max,
        #         left_max / right_min, left_max / right_max]
        #     new_range = min(values), max(values)
        #     return expr.update_range(*new_range)
        case ir.Binary(left, right, ir.BinOp.Min):
            left_min, left_max = left.range
            right_min, right_max = right.range
            return expr.update_range(min(left_min, right_min), min(left_max, right_max))
        case ir.Binary(left, right, ir.BinOp.Max):
            left_min, left_max = left.range
            right_min, right_max = right.range
            return expr.update_range(max(left_min, right_min), max(left_max, right_max))
        case ir.Unary(orig, ir.UnOp.Sqrt, _):
            orig_min, orig_max = orig.range
            return expr.update_range(0, absint_sqrt(orig_max))
        case ir.Unary(orig, ir.UnOp.Abs, _):
            orig_min, orig_max = orig.range
            abs_min = min(absint_abs(orig_min), absint_abs(orig_max))
            abs_max = max(absint_abs(orig_min), absint_abs(orig_max))
            assert abs_min >= 0
            return expr.update_range(abs_min, abs_max)
        case ir.Unary(orig, ir.UnOp.Cos, _) | ir.Unary(orig, ir.UnOp.Sin, _):
            # TODO(max): Be more precise using input range
            return expr.update_range(-1, 1)
        case ir.Unary(orig, ir.UnOp.Sign, _):
            return expr.update_range(-1, 1)
        case ir.Broadcast(orig, _):
            # TODO(max): More complex value that indicates that this is an
            # array
            return expr.update_range(*orig.range)
        case ir.ComparisonIf(a, b, ctrue, cfalse, _):
            ctrue_min, ctrue_max = ctrue.range
            cfalse_min, cfalse_max = cfalse.range
            return expr.update_range(
                min(ctrue_min, cfalse_min), max(ctrue_max, cfalse_max)
            )
        case _:
            return False


def cse(expr: ir.Expr) -> bool:
    seen = {}
    changed = False
    exprs = topo(expr)
    for e in exprs:
        # Force recomputation of hash_value because forwarded nodes might have
        # changed; gets more aggressive CSE
        e.__dict__.pop("hash_value", None)
    for e in exprs:
        if e in seen:
            e.make_equal_to(seen[e])
            changed = True
        else:
            seen[e] = e
    return changed
