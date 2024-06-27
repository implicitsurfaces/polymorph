from dataclasses import dataclass
from enum import Enum

import jax.numpy as jnp


def as_expr(x):
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


class UnOp(Enum):
    Sqrt = "sqrt"
    Sigmoid = "sigmoid"
    SmoothAbs = "smoothabs"
    SoftPlus = "softplus"


class Expr:
    dim: int

    def __init__(self, dim):
        object.__setattr__(self, "dim", dim)

    def __mul__(self, other):
        return self.__binary(other, BinOp.Mul)

    def __add__(self, other):
        return self.__binary(other, BinOp.Add)

    def __sub__(self, other):
        return self.__binary(other, BinOp.Sub)

    def __truediv__(self, other):
        return self.__binary(other, BinOp.Div)

    def __pow__(self, exponent):
        return self.__binary(exponent, BinOp.Exp)

    def __neg__(self):
        return self.__binary(-1, BinOp.Mul)

    def __binary(self, other, op):
        o = as_expr(other)
        if self.dim == o.dim:
            return Binary(self, o, op)
        elif self.dim == 1:
            return Binary(Broadcast(self, o.dim), o, op)
        elif o.dim == 1:
            return Binary(self, Broadcast(o, self.dim), op)

        raise ValueError()

    def sqrt(self):
        return Unary(self, UnOp.Sqrt)

    def sigmoid(self):
        return Unary(self, UnOp.Sigmoid)

    def smoothabs(self):
        return Unary(self, UnOp.SmoothAbs)

    def softplus(self):
        return Unary(self, UnOp.SoftPlus)


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
    value: list[float]

    def __post_init__(self):
        super().__init__(len(self.value))


@dataclass(frozen=True)
class Binary(Expr):
    left: Expr
    right: Expr
    op: BinOp

    def __post_init__(self):
        if self.left.dim == self.right.dim:
            super().__init__(self.left.dim)
        else:
            raise ValueError()


@dataclass(frozen=True)
class Unary(Expr):
    orig: Expr
    op: UnOp

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


@dataclass(frozen=True)
class Sum(Expr):
    orig: Expr

    def __post_init__(self):
        super().__init__(1)


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
