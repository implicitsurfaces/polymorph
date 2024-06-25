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
    elif isinstance(x, jnp.ndarray) and len(x.shape) == 1:
        return Arr(x)

    raise ValueError(f"{x} ({type(x)})")


class BinOp(Enum):
    Mul = "mul"
    Add = "add"
    Div = "div"
    Sub = "sub"
    Exp = "exp"


class UnOp(Enum):
    Sqrt = "sqrt"
    Sigmoid = "sigmoid"
    SmoothAbs = "smoothabs"


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
class Broadcast(Expr):
    orig: Expr
    dim: int


@dataclass(frozen=True)
class Sum(Expr):
    orig: Expr

    def __post_init__(self):
        super().__init__(1)
