from polymorph_num import ops
from polymorph_num.expr import ONE, PI, SQRT2_INV, ZERO, Expr
from polymorph_num.vec import Vec2


def negative_sign(val: Expr) -> Expr:
    return ops.if_lt(val, 0, -1, 1)


class Angle:
    _cos: Expr
    _sin: Expr

    def __init__(self, cos: Expr, sin: Expr):
        self._cos = cos
        self._sin = sin

    def __add__(self, other: "Angle") -> "Angle":
        return Angle(
            self._cos * other._cos - self._sin * other._sin,
            self._sin * other._cos + self._cos * other._sin,
        )

    def __sub__(self, other: "Angle") -> "Angle":
        return Angle(
            self._cos * other._cos + self._sin * other._sin,
            self._sin * other._cos - self._cos * other._sin,
        )

    def __neg__(self) -> "Angle":
        return Angle(self._cos, -self._sin)

    def cos(self) -> Expr:
        return self._cos

    def sin(self) -> Expr:
        return self._sin

    def tan(self) -> Expr:
        return self._sin / self._cos

    def half(self) -> "Angle":
        return Angle(
            negative_sign(self._sin) * ((1 + self._cos) / 2).sqrt(),
            ((1 - self._cos) / 2).sqrt(),
        )

    def double(self) -> "Angle":
        return Angle(
            self._cos * self._cos - self._sin * self._sin, 2 * self._cos * self._sin
        )

    def as_vec(self) -> Vec2:
        return Vec2(self._cos, self._sin)

    def as_rad(self) -> Expr:
        # I have tried to use acos, but this is more nurmerically stable
        return ops.atan2(self._sin, self._cos)

    def as_deg(self) -> Expr:
        return self.as_rad() * 180 / PI

    def as_sort_value(self) -> Expr:
        return ops.if_lt(self._sin.sign(), 0, 3 + self._cos, 1 - self._cos) / 2

    def perp(self) -> "Angle":
        return Angle(-self._sin, self._cos)

    def flip_sign(self, orientation_sign: Expr) -> "Angle":
        """Changes the angle to the opposite direction depending on the orientation sign.

        Note that the orientation sign must be 1 or -1.
        """

        return Angle(self._cos, self._sin * orientation_sign)

    def quarter_turn(self) -> Expr:
        # There should be an issue with this function when the angle is exactly 0
        sign_cos = self._cos.sign()
        sign_sin = self._sin.sign()

        return (ONE - sign_sin) + (ONE - sign_cos) * (ONE + sign_sin) / 2

    def quadrant(self):
        q1_q3 = (self._sin * self._cos).sign()
        q1_q2 = self._sin.sign()

        correction_sin_0 = ops.if_eq(self._sin, 0, ops.if_gt(self._cos, 0, -3, 1), 0)
        correction_cos_0 = ops.if_eq(self._cos, 0, 1, 0)

        return (3 - 2 * q1_q2 - q1_q3 + correction_sin_0 + correction_cos_0) / 2


def polar_angle(x: Expr, y: Expr) -> Angle:
    r = (x * x + y * y).sqrt()
    return Angle(x / r, y / r)


def polar_angle_from_vec(v: Vec2) -> Angle:
    return polar_angle(v.x, v.y)


def angle_from_rad(rad: Expr) -> Angle:
    return Angle(rad.cos(), rad.sin())


def angle_from_deg(deg: Expr) -> Angle:
    return angle_from_rad(deg * PI / 180)


def angle_from_sin(sin: Expr) -> Angle:
    cos = (1.0 - (sin * sin)).sqrt()
    return Angle(cos, sin)


def angle_from_cos(cos: Expr) -> Angle:
    sin = (1.0 - (cos * cos)).sqrt()
    return Angle(cos, sin)


def two_vectors_angle(u: Vec2, v: Vec2) -> Angle:
    norm_u = u.norm()
    norm_v = v.norm()

    unit_u = u / norm_u
    unit_v = v / norm_v

    sin = unit_u.cross(unit_v)
    cos = unit_u.dot(unit_v)

    return Angle(cos, sin)


NO_TURN = Angle(ONE, ZERO)
HALF_TURN = Angle(-ONE, ZERO)
FULL_TURN = Angle(ONE, ZERO)
QUARTER_TURN = Angle(ZERO, ONE)
THREE_QUARTER_TURN = Angle(ZERO, -ONE)
EIGTH_TURN = Angle(SQRT2_INV, SQRT2_INV)
