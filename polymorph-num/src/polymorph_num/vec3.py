from dataclasses import dataclass

from polymorph_num.vec import Vec2

from .expr import Expr, Num, as_expr


@dataclass(init=False, frozen=True)
class Vec3:
    x: Expr
    y: Expr
    z: Expr

    @classmethod
    def from_vec2(cls, v: Vec2):
        return cls(v.x, v.y, 0.0)

    def __init__(self, x: Num, y: Num, z: Num):
        object.__setattr__(self, "x", as_expr(x))
        object.__setattr__(self, "y", as_expr(y))
        object.__setattr__(self, "z", as_expr(z))

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __add__(self, other):
        match other:
            case Vec3(x, y, z):
                return Vec3(self.x + x, self.y + y, self.z + z)
            case _:
                raise NotImplementedError()

    def __truediv__(self, other):
        return Vec3(
            self.x / as_expr(other), self.y / as_expr(other), self.z / as_expr(other)
        )

    def __neg__(self):
        return Vec3(-self.x, -self.y, -self.z)

    def norm(self):
        return (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()

    def scale(self, other):
        return Vec3(self.x * other, self.y * other, self.z * other)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def norm_squared(self):
        return self.x * self.x + self.y * self.y + self.z * self.z

    def normalize(self):
        return self / self.norm()

    def rotate(self, angle: Num, axis: "Vec3"):
        """
        Uses the Rodrigues' rotation formula. See:
        https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

        :param angle: The angle to rotate by.
        :param axis: The axis to rotate around. Note that this vector should be normalized.
        """
        angle = as_expr(angle)

        c = angle.cos()
        s = angle.sin()

        return (
            self.scale(c)
            + axis.scale(axis.dot(self) * (1 - c))
            + axis.cross(self).scale(s)
        )

    def translateTo(self, other: "Vec3"):
        return self - other

    @property
    def dim(self):
        return self.x.dim


type ValVec3 = tuple[Num, Num, Num] | Vec2 | Vec3


def as_vec3(p: ValVec3) -> Vec3:
    if isinstance(p, Vec3):
        return p
    if isinstance(p, Vec2):
        return Vec3.from_vec2(p)
    return Vec3(p[0], p[1], p[2])


ORIGIN = Vec3(0, 0, 0)
X_AXIS = Vec3(1, 0, 0)
Y_AXIS = Vec3(0, 1, 0)
Z_AXIS = Vec3(0, 0, 1)
