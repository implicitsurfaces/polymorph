from __future__ import annotations
import math
import dataclasses
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Sequence

import jax.numpy as jnp

type Num = int | float | "Expr" | jnp.ndarray


def as_expr(x: Num) -> "Expr":
    if isinstance(x, Expr):
        return x
    elif isinstance(x, float):
        return Scalar(x)
    elif isinstance(x, int):
        return Scalar(float(x))
    elif isinstance(x, jnp.ndarray):
        assert len(x.shape) == 1, f"Expected 1D array, got {x.shape}"
        return Arr(x)

    raise ValueError(f"{x} ({type(x)})")


class BinOp(Enum):
    Mul = "mul"
    Add = "add"
    Div = "div"
    Sub = "sub"
    Exp = "exp"
    Max = "max"
    Min = "min"
    Mod = "mod"
    ArcTan2 = "arctan2"
    RMS = "rms"


class UnOp(Enum):
    Sqrt = "sqrt"
    Sigmoid = "sigmoid"
    SmoothAbs = "smoothabs"
    Abs = "abs"
    SoftPlus = "softplus"
    Log = "log"
    Exp = "exp"
    Cos = "cos"
    Sin = "sin"
    Tanh = "tanh"
    ArcTan = "arctan"
    Sign = "sign"
    Boxcar = "boxcar"
    Sqr = "sqr"


class ComparisonOp(Enum):
    Gt = "gt"
    Ge = "ge"
    Eq = "eq"


def broadcast_binary(left, right, op):
    if left.dim == right.dim:
        return Binary(left, right, op)
    elif left.dim == 1:
        return Binary(Broadcast(left, right.dim), right, op)
    elif right.dim == 1:
        return Binary(left, Broadcast(right, left.dim), op)

    raise ValueError(f"Dimension mismatch: {left.dim} != {right.dim}")


def broacast_args(*args):
    dims = set(arg.dim for arg in args)

    if len(dims) == 1:
        return args

    if len(dims) > 2 or min(dims) != 1:
        raise ValueError(f"Dimension mismatch, multiple dimensions: {dims}")

    big_dim = max(dims)
    return [Broadcast(arg, big_dim) if arg.dim == 1 else arg for arg in args]


class Expr:
    dim: int
    forwarded: Expr | None

    def find(self):
        expr = self
        while expr is not None:
            next = expr.forwarded
            if next is None:
                return expr
            expr = next
        return expr

    def make_equal_to(self, other):
        found = self.find()
        if found is not other:
            found.set_forwarded(other)

    def set_forwarded(self, other):
        object.__setattr__(self, "forwarded", other)

    def __init__(self, dim: int):
        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "forwarded", None)

    def __mul__(self, other: Num):
        return broadcast_binary(self, as_expr(other), BinOp.Mul)

    def __rmul__(self, other: Num):
        return broadcast_binary(as_expr(other), self, BinOp.Mul)

    def __add__(self, other: Num):
        return broadcast_binary(self, as_expr(other), BinOp.Add)

    def __radd__(self, other: Num):
        return broadcast_binary(as_expr(other), self, BinOp.Add)

    def __sub__(self, other: Num):
        return broadcast_binary(self, as_expr(other), BinOp.Sub)

    def __rsub__(self, other: Num):
        return broadcast_binary(as_expr(other), self, BinOp.Sub)

    def __truediv__(self, other: Num):
        return broadcast_binary(self, as_expr(other), BinOp.Div)

    def __rtruediv__(self, other: Num):
        return broadcast_binary(as_expr(other), self, BinOp.Div)

    def __pow__(self, exponent: Num):
        return broadcast_binary(self, as_expr(exponent), BinOp.Exp)

    def __rpow__(self, base: Num):
        return broadcast_binary(as_expr(base), self, BinOp.Exp)

    def __mod__(self, mod: Num):
        return broadcast_binary(self, as_expr(mod), BinOp.Mod)

    def __rmod__(self, mod: Num):
        return broadcast_binary(as_expr(mod), self, BinOp.Mod)

    def __neg__(self: Num):
        return broadcast_binary(self, as_expr(-1), BinOp.Mul)

    def sqrt(self):
        return Unary(self, UnOp.Sqrt)

    def sigmoid(self):
        return Unary(self, UnOp.Sigmoid)

    def smoothabs(self):
        return Unary(self, UnOp.SmoothAbs)

    def abs(self):
        return Unary(self, UnOp.Abs)

    def softplus(self):
        return Unary(self, UnOp.SoftPlus)

    def softminus(self):
        return self - Unary(self, UnOp.SoftPlus)

    def log(self):
        return Unary(self, UnOp.Log)

    def exp(self):
        return Unary(self, UnOp.Exp)

    def cos(self):
        return Unary(self, UnOp.Cos)

    def sin(self):
        return Unary(self, UnOp.Sin)

    def atan(self):
        return Unary(self, UnOp.ArcTan)

    def sign(self):
        return Unary(self, UnOp.Sign)

    def tanh(self):
        return Unary(self, UnOp.Tanh)

    def boxcar(self, min: float, max: float):
        return Unary(self, UnOp.Boxcar, (min, max))


@dataclass(frozen=True)
class Param(Expr):
    id: int

    def __post_init__(self):
        super().__init__(1)


@dataclass(frozen=True)
class Observation(Expr):
    name: str

    def __post_init__(self):
        super().__init__(1)


@dataclass(frozen=True)
class Scalar(Expr):
    value: float

    def __post_init__(self):
        super().__init__(1)


@dataclass(frozen=True)
class Arr(Expr):
    value: Sequence[float] | jnp.ndarray

    def __post_init__(self):
        super().__init__(len(self.value))

    # We need to implement __hash__ so instances can be used as dict keys.
    # `list` is not hashable, and we don't want to convert it to a tuple
    # because we've empirically observed this to be quite expensive.
    def __hash__(self):
        return id(self)


@dataclass(frozen=True)
class Binary(Expr):
    left: Expr
    right: Expr
    op: BinOp

    __match_args__ = ("left_found", "right_found", "op")

    @cached_property
    def hash_value(self) -> int:
        return hash((self.dim, self.left, self.right, self.op))

    @property
    def left_found(self): return self.left.find()

    @property
    def right_found(self): return self.right.find()

    def __hash__(self):
        return self.hash_value

    def __post_init__(self):
        if self.left.dim == self.right.dim:
            super().__init__(self.left.dim)
        else:
            raise ValueError(f"Dimension mismatch: {self.left.dim} != {self.right.dim}")


@dataclass(frozen=True)
class Unary(Expr):
    orig: Expr
    op: UnOp
    constants: tuple = ()

    __match_args__ = ("orig_found", "op", "constants")

    @property
    def orig_found(self): return self.orig.find()

    @cached_property
    def hash_value(self) -> int:
        return hash((self.dim, self.orig, self.op, self.constants))

    def __hash__(self):
        return self.hash_value

    def __post_init__(self):
        super().__init__(self.orig.dim)


@dataclass(frozen=True)
class Debug(Expr):
    tag: str
    orig: Expr

    def __post_init__(self):
        super().__init__(self.orig.dim)


@dataclass(frozen=True)
class Broadcast(Expr):
    orig: Expr
    dim: int

    def __post_init__(self):
        super().__init__(self.dim)


@dataclass(frozen=True)
class Sum(Expr):
    orig: Expr

    def __post_init__(self):
        super().__init__(1)


@dataclass(frozen=True)
class GridX(Expr):
    width: int
    height: int

    def __post_init__(self):
        super().__init__(self.width * self.height)


@dataclass(frozen=True)
class GridY(Expr):
    width: int
    height: int

    def __post_init__(self):
        super().__init__(self.width * self.height)


@dataclass(frozen=True)
class GridX3d(Expr):
    width: int
    height: int
    depth: int

    def __post_init__(self):
        super().__init__(self.width * self.height * self.depth)


@dataclass(frozen=True)
class GridY3d(Expr):
    width: int
    height: int
    depth: int

    def __post_init__(self):
        super().__init__(self.width * self.height * self.depth)


@dataclass(frozen=True)
class GridZ3d(Expr):
    width: int
    height: int
    depth: int

    def __post_init__(self):
        super().__init__(self.width * self.height * self.depth)


@dataclass(frozen=True)
class Random(Expr):
    dim: int
    low: Expr
    high: Expr

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return id(self) == id(other)


@dataclass(frozen=True)
class ComparisonIf(Expr):
    a: Expr
    b: Expr
    condition_true: Expr
    condition_false: Expr
    op: ComparisonOp

    __match_args__ = ("a_found", "b_found", "condition_true_found", "condition_false_found", "op")

    @property
    def a_found(self): return self.a.find()

    @property
    def b_found(self): return self.b.find()

    @property
    def condition_true_found(self): return self.condition_true.find()

    @property
    def condition_false_found(self): return self.condition_false.find()

    @cached_property
    def hash_value(self):
        return hash(
            (
                self.dim,
                self.a,
                self.b,
                self.condition_true,
                self.condition_false,
                self.op,
            )
        )

    def __hash__(self):
        return self.hash_value

    def __post_init__(self):
        if (
            len(
                {
                    self.a.dim,
                    self.b.dim,
                    self.condition_true.dim,
                    self.condition_false.dim,
                }
            )
            != 1
        ):
            raise ValueError("Dimension mismatch")

        super().__init__(self.a.dim)


def to_str(expr: Expr, indent: str = "") -> str:
    if isinstance(expr, Scalar):
        return f"{indent}Scalar({expr.value})"
    elif isinstance(expr, Arr):
        return f"{indent}Arr({expr.value})"
    elif isinstance(expr, Param):
        return f"{indent}Param({expr.id})"
    elif isinstance(expr, Observation):
        return f"{indent}Observation('{expr.name}')"
    elif isinstance(expr, Binary):
        return (
            f"{indent}Binary({expr.op},\n"
            f"{to_str(expr.left, indent + '  ')},\n"
            f"{to_str(expr.right, indent + '  ')},\n"
            f"{indent})"
        )
    elif isinstance(expr, Unary):
        return (
            f"{indent}Unary({expr.op},\n"
            f"{to_str(expr.orig, indent + '  ')},\n"
            f"{indent})"
        )
    elif isinstance(expr, Broadcast):
        return (
            f"{indent}Broadcast(\n"
            f"{to_str(expr.orig, indent + '  ')},\n"
            f"{indent}  {expr.dim}\n"
            f"{indent})"
        )
    elif isinstance(expr, Sum):
        return f"{indent}Sum(\n" f"{to_str(expr.orig, indent + '  ')}\n" f"{indent})"
    else:
        return f"{indent}Unknown Expr type: {type(expr)}"


Infinity = Scalar(float("inf"))
PI = Scalar(math.pi)
TAU = Scalar(2 * math.pi)
ZERO = Scalar(0.0)
ONE = Scalar(1.0)
