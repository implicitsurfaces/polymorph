from polymorph_num.angle import Angle
from polymorph_num.expr import Expr
from polymorph_num.vec import Vec2


class Length:
    value: Expr

    def __init__(self, value: Expr):
        self.value = value

    def as_vec(self, angle: Angle) -> Vec2:
        return Vec2(angle.cos() * self.value, angle.sin() * self.value)
