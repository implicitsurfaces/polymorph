from polymorph_num.expr import ONE, PI, SQRT2_INV, ZERO, Expr
from polymorph_num.vec import Vec2


def polar_diamond_angle(x, y):
    sign_x = x.sign()
    sign_y = y.sign()

    x = x.abs()
    y = y.abs()

    denom = x + y

    q1 = y / denom
    q2 = 1 + x / denom
    q3 = 2 + y / denom
    q4 = 3 + x / denom

    is_q1 = (1 + sign_y) * (1 + sign_x)
    is_q2 = (1 + sign_y) * (1 - sign_x)
    is_q3 = (1 - sign_y) * (1 - sign_x)
    is_q4 = (1 - sign_y) * (1 + sign_x)

    return 0.25 * (q1 * is_q1 + q2 * is_q2 + q3 * is_q3 + q4 * is_q4)


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
            self._sin.sign() * ((1 + self._cos) / 2).sqrt(),
            ((1 - self._cos) / 2).sqrt(),
        )

    def double(self) -> "Angle":
        return Angle(
            self._cos * self._cos - self._sin * self._sin, 2 * self._cos * self._sin
        )

    def as_vec(self) -> Vec2:
        return Vec2(self._cos, self._sin)

    def as_rad(self) -> Expr:
        return self._cos.acos() * self._sin.sign()

    def as_deg(self) -> Expr:
        return self.as_rad() * 180 / PI

    def as_diamond_angle(self) -> Expr:
        return polar_diamond_angle(self._cos, self._sin)

    def perp(self) -> "Angle":
        return Angle(-self._sin, self._cos)

    def quarter_turn(self) -> Expr:
        # There should be an issue with this function when the angle is exactly 0
        sign_cos = self._cos.sign()
        sign_sin = self._sin.sign()

        return (ONE - sign_sin) + (ONE - sign_cos) * (ONE + sign_sin) / 2


def polar_angle(x: Expr, y: Expr) -> Angle:
    r = (x * x + y * y).sqrt()
    return Angle(x / r, y / r)


def angle_from_rad(rad: Expr) -> Angle:
    return Angle(rad.cos(), rad.sin())


def angle_from_deg(deg: Expr) -> Angle:
    return angle_from_rad(deg * PI / 180)


def two_vectors_angle(u: Vec2, v: Vec2) -> Angle:
    norm_u = u.norm()
    norm_v = v.norm()
    norm = norm_u * norm_v

    sin = u.cross(v) / norm
    cos = u.dot(v) / norm

    return Angle(cos, sin)


HALF_TURN = Angle(-ONE, ZERO)
FULL_TURN = Angle(ONE, ZERO)
QUARTER_TURN = Angle(ZERO, ONE)
THREE_QUARTER_TURN = Angle(ZERO, -ONE)
EIGTH_TURN = Angle(SQRT2_INV, SQRT2_INV)
