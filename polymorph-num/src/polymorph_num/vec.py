from dataclasses import dataclass

from .expr import Expr, Num, as_expr


@dataclass(init=False, frozen=True)
class Vec2:
    x: Expr
    y: Expr

    @classmethod
    def origin(cls):
        return cls(0.0, 0.0)

    def __init__(self, x: Num, y: Num):
        object.__setattr__(self, "x", as_expr(x))
        object.__setattr__(self, "y", as_expr(y))

    def __sub__(self, other):
        return Vec2(self.x - other.x, self.y - other.y)

    def __add__(self, other):
        match other:
            case Vec2(x, y):
                return Vec2(self.x + x, self.y + y)
            case _:
                raise NotImplementedError()

    def __truediv__(self, other):
        return Vec2(self.x / as_expr(other), self.y / as_expr(other))

    def __neg__(self):
        return Vec2(-self.x, -self.y)

    def norm(self):
        return (self.x * self.x + self.y * self.y).sqrt()

    def scale(self, other):
        return Vec2(self.x * other, self.y * other)

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def cross(self, other):
        return self.x * other.y - self.y * other.x

    def norm_squared(self):
        return self.x * self.x + self.y * self.y

    @property
    def dim(self):
        return self.x.dim


type ValVec = tuple[Num, Num] | Vec2


def as_vec2(p: ValVec):
    if isinstance(p, Vec2):
        return p
    return Vec2(p[0], p[1])
