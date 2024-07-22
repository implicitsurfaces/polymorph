from polymorph_num import ops
from polymorph_num.expr import ZERO, Expr, Num, as_expr

from polymorph_s2df.bounding_box import BoundingBox

from .operations import Shape
from .utils import norm

NEAR_INFINITY = as_expr(1e9)


class BottomHalfPlane(Shape):
    def __init__(self):
        pass

    def __eq__(self, other):
        return isinstance(other, BottomHalfPlane)

    def __hash__(self):
        return hash("BottomHalfPlane")

    def __repr__(self):
        return "BottomHalfPlane()"

    def distance(self, x: Num, y: Num) -> Expr:
        return as_expr(y)

    def bounding_box(self):
        return BoundingBox(-NEAR_INFINITY, -NEAR_INFINITY, NEAR_INFINITY, ZERO)


class TopHalfPlane(Shape):
    def __init__(self):
        pass

    def __eq__(self, other):
        return isinstance(other, TopHalfPlane)

    def __hash__(self):
        return hash("TopHalfPlane")

    def __repr__(self):
        return "TopHalfPlane()"

    def distance(self, x: Num, y: Num) -> Expr:
        return as_expr(-y)

    def bounding_box(self):
        return BoundingBox(-NEAR_INFINITY, ZERO, NEAR_INFINITY, NEAR_INFINITY)


class LeftHalfPlane(Shape):
    def __init__(self):
        pass

    def __eq__(self, other):
        return isinstance(other, LeftHalfPlane)

    def __hash__(self):
        return hash("LeftHalfPlane")

    def __repr__(self):
        return "LeftHalfPlane()"

    def distance(self, x: Num, y: Num) -> Expr:
        return as_expr(x)

    def bounding_box(self):
        return BoundingBox(ZERO, -NEAR_INFINITY, NEAR_INFINITY, NEAR_INFINITY)


class RightHalfPlane(Shape):
    def __init__(self):
        pass

    def __eq__(self, other):
        return isinstance(other, RightHalfPlane)

    def __hash__(self):
        return hash("RightHalfPlane")

    def __repr__(self):
        return "RightHalfPlane()"

    def distance(self, x: Num, y: Num) -> Expr:
        return as_expr(-x)

    def bounding_box(self):
        return BoundingBox(-NEAR_INFINITY, -NEAR_INFINITY, ZERO, NEAR_INFINITY)


class Circle(Shape):
    def __init__(self, radius: Num) -> None:
        super().__init__()
        self.radius = as_expr(radius)

    def astuple(self):
        return (self.radius,)

    def __eq__(self, other):
        if not isinstance(other, Circle):
            return False
        return self.astuple() == other.astuple()

    def __hash__(self):
        return hash(self.astuple())

    def __repr__(self):
        return f"Circle({self.radius})"

    def distance(self, x: Num, y: Num) -> Expr:
        x = as_expr(x)
        y = as_expr(y)

        return (x * x + y * y).sqrt() - self.radius

    def bounding_box(self):
        return BoundingBox(-self.radius, -self.radius, self.radius, self.radius)


class Box(Shape):
    def __init__(self, width, height):
        super().__init__()
        self.width = as_expr(width)
        self.height = as_expr(height)

    def astuple(self):
        return (self.width, self.height)

    def __eq__(self, other):
        if not isinstance(other, Box):
            return False
        return self.astuple() == other.astuple()

    def __hash__(self):
        return hash(self.astuple())

    def __repr__(self):
        return f"Box({self.width}, {self.height})"

    def distance(self, x: Num, y: Num) -> Expr:
        q_x = as_expr(x).smoothabs() - self.width / 2
        q_y = as_expr(y).smoothabs() - self.height / 2

        return norm(q_x.softplus(), q_y.softplus()) + ops.max(q_x, q_y).softminus()

    def bounding_box(self):
        return BoundingBox(
            -self.width / 2, -self.height / 2, self.width / 2, self.height / 2
        )
