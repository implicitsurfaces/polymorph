from polymorph_num import ops
from polymorph_num.expr import Expr, Num, as_expr
from polymorph_num.vec import ValVec, as_vec2

from .utils import indent_shape


class Shape:
    def distance(self, x: Num, y: Num) -> Expr:
        raise NotImplementedError

    def is_inside(self, x: Num, y: Num, scale=100) -> Expr:
        return 1 - (self.distance(x, y) * scale).sigmoid()

    def is_on_boundary(self, x: Num, y: Num, dist=2.0) -> Expr:
        return self.distance(x, y).boxcar(-dist / 2.0, dist / 2.0)

    def area(self, sample_xs: Num, sample_ys: Num, sample_area: Num):
        return ops.mean(self.is_inside(sample_xs, sample_ys)) * sample_area

    def astuple(self):
        raise NotImplementedError

    def __hash__(self):
        return hash(self.astuple())

    def union(self, other: "Shape"):
        return Union(self, other)

    def intersect(self, other: "Shape"):
        return Intersection(self, other)

    def substract(self, other: "Shape"):
        return Substraction(self, other)

    def smooth_union(self, k: Num, other: "Shape"):
        return SmoothUnion(k, self, other)

    def smooth_intersect(self, k: Num, other: "Shape"):
        return SmoothIntersection(k, self, other)

    def smooth_substract(self, k, other: "Shape"):
        return SmoothSubstraction(k, self, other)

    def shell(self, thickness: Num):
        return Shell(thickness, self)

    def translate(self, offset: ValVec):
        return Translation(offset, self)

    def rotate(self, angle: Num):
        return Rotation(angle, self)

    def rotate_around(self, angle: Num, center: ValVec):
        return Translation(center, Rotation(angle, Translation(-as_vec2(center), self)))

    def scale(self, factor: Num):
        return Scale(factor, self)

    def invert(self):
        return Inversion(self)

    def dilate(self, offset: Num):
        return Dilate(offset, self)

    def morph(self, t: Num, other: "Shape"):
        return Morph(t, self, other)

    def taper(self, height: Num, factor: Num):
        return Taper(height, factor, self)

    def taper_from_point(self, point: ValVec, height: Num, factor: Num):
        return Translation(
            point, Taper(height, factor, Translation(-as_vec2(point), self))
        )


class Translation(Shape):
    def __init__(self, offset: ValVec, shape: Shape):
        super().__init__()
        self.offset = as_vec2(offset)
        self.shape = shape

    def astuple(self):
        return self.offset, self.shape

    def __hash__(self):
        return hash(self.astuple())

    def __eq__(self, other):
        if not isinstance(other, Translation):
            return False
        return self.astuple() == other.astuple()

    def __repr__(self):
        return f"Translation(\n  {self.offset},\n{indent_shape(self.shape)}\n)"

    def distance(self, x: Num, y: Num) -> Expr:
        return self.shape.distance(x - self.offset.x, y - self.offset.y)


class Rotation(Shape):
    def __init__(self, angle: Num, shape: Shape):
        super().__init__()
        self.angle = as_expr(angle)
        self.shape = shape

    def astuple(self):
        return self.angle, self.shape

    def __hash__(self):
        return hash(self.astuple())

    def __eq__(self, other):
        if not isinstance(other, Rotation):
            return False
        return self.astuple() == other.astuple()

    def __repr__(self):
        return f"Rotation(\n  {self.angle}, \n{indent_shape(self.shape)}\n)"

    def distance(self, x: Num, y: Num) -> Expr:
        x = as_expr(x)
        y = as_expr(y)

        c = self.angle.cos()
        s = -self.angle.sin()

        x2 = (x * c) - (y * s)
        y2 = (x * s) + (y * c)

        return self.shape.distance(x2, y2)


class Intersection(Shape):
    def __init__(self, shape_1: Shape, shape_2: Shape):
        super().__init__()
        self.shape_1 = shape_1
        self.shape_2 = shape_2

    def astuple(self):
        return self.shape_1, self.shape_2

    def __hash__(self):
        return hash(self.astuple())

    def __eq__(self, other):
        if not isinstance(other, Intersection):
            return False
        return self.astuple() == other.astuple()

    def __repr__(self):
        return f"Intersection(\n{indent_shape(self.shape_1)}, \n{indent_shape(self.shape_2)}\n)"

    def distance(self, x: Num, y: Num) -> Expr:
        return ops.max(self.shape_1.distance(x, y), self.shape_2.distance(x, y))


class SmoothIntersection(Shape):
    def __init__(self, k: Num, shape_1: Shape, shape_2: Shape):
        super().__init__()
        self.k = as_expr(k * 4)
        self.shape_1 = shape_1
        self.shape_2 = shape_2

    def astuple(self):
        return self.k, self.shape_1, self.shape_2

    def __hash__(self):
        return hash(self.astuple())

    def __eq__(self, other):
        if not isinstance(other, SmoothIntersection):
            return False
        return self.astuple() == other.astuple()

    def __repr__(self):
        return f"SmoothIntersection(\n  {self.k / 4},\n{indent_shape(self.shape_1)}, \n{indent_shape(self.shape_2)} \n)"

    def distance(self, x: Num, y: Num) -> Expr:
        val1 = self.shape_1.distance(x, y)
        val2 = self.shape_2.distance(x, y)

        h = ops.max(self.k - (val1 - val2).abs(), 0.0) / self.k
        return ops.max(val1, val2) + h * h * self.k * 1.0 / 4.0


class Union(Shape):
    def __init__(self, shape_1: Shape, shape_2: Shape):
        super().__init__()
        self.shape_1 = shape_1
        self.shape_2 = shape_2

    def astuple(self):
        return self.shape_1, self.shape_2

    def __hash__(self):
        return hash(self.astuple())

    def __eq__(self, other):
        if not isinstance(other, Union):
            return False
        return self.astuple() == other.astuple()

    def __repr__(self):
        return (
            f"Union(\n{indent_shape(self.shape_1)}, \n{indent_shape(self.shape_2)}\n)"
        )

    def distance(self, x: Num, y: Num) -> Expr:
        return ops.min(self.shape_1.distance(x, y), self.shape_2.distance(x, y))


class SmoothUnion(Shape):
    def __init__(self, k, shape_1: Shape, shape_2: Shape):
        super().__init__()
        self.k = as_expr(k * 4)
        self.shape_1 = shape_1
        self.shape_2 = shape_2

    def astuple(self):
        return self.k, self.shape_1, self.shape_2

    def __hash__(self):
        return hash(self.astuple())

    def __eq__(self, other):
        if not isinstance(other, SmoothUnion):
            return False
        return self.astuple() == other.astuple()

    def __repr__(self):
        return f"SmoothUnion(\n  {self.k / 4},\n{indent_shape(self.shape_1)}, \n{indent_shape(self.shape_2)} \n)"

    def distance(self, x: Num, y: Num) -> Expr:
        val1 = self.shape_1.distance(x, y)
        val2 = self.shape_2.distance(x, y)

        h = ops.max(self.k - (val1 - val2).abs(), 0.0) / self.k
        return ops.min(val1, val2) - h * h * self.k * 1.0 / 4.0


class Substraction(Shape):
    def __init__(self, shape_1: Shape, shape_2: Shape):
        super().__init__()
        self.shape_1 = shape_1
        self.shape_2 = shape_2

    def astuple(self):
        return self.shape_1, self.shape_2

    def __hash__(self):
        return hash(self.astuple())

    def __eq__(self, other):
        if not isinstance(other, Substraction):
            return False
        return self.astuple() == other.astuple()

    def __repr__(self):
        return f"Substraction(\n{indent_shape(self.shape_1)}, \n{indent_shape(self.shape_2)}\n)"

    def distance(self, x: Num, y: Num) -> Expr:
        return ops.max(self.shape_1.distance(x, y), -self.shape_2.distance(x, y))


class SmoothSubstraction(Shape):
    def __init__(self, k: Num, shape_1: Shape, shape_2: Shape):
        super().__init__()
        self.k = as_expr(k * 4)
        self.shape_1 = shape_1
        self.shape_2 = shape_2

    def astuple(self):
        return self.k, self.shape_1, self.shape_2

    def __hash__(self):
        return hash(self.astuple())

    def __eq__(self, other):
        if not isinstance(other, SmoothSubstraction):
            return False
        return self.astuple() == other.astuple()

    def __repr__(self):
        return f"SmoothSubstraction(\n  {self.k / 4},\n{indent_shape(self.shape_1)}, \n{indent_shape(self.shape_2)} \n)"

    def distance(self, x: Num, y: Num) -> Expr:
        val1 = self.shape_1.distance(x, y)
        val2 = self.shape_2.distance(x, y)

        h = ops.max(self.k - (val1 + val2).abs(), 0.0) / self.k
        return ops.max(val1, -val2) + h * h * self.k * (1.0 / 4.0)


class Scale(Shape):
    def __init__(self, factor: Num, shape: Shape):
        super().__init__()
        self.factor = as_expr(factor)
        self.shape = shape

    def astuple(self):
        return self.factor, self.shape

    def __hash__(self):
        return hash(self.astuple())

    def __eq__(self, other):
        if not isinstance(other, Scale):
            return False
        return self.astuple() == other.astuple()

    def __repr__(self):
        return f"Scale(\n  {self.factor},\n{indent_shape(self.shape)}\n)"

    def distance(self, x: Num, y: Num) -> Expr:
        return (
            self.shape.distance(as_expr(x) / self.factor, as_expr(y) / self.factor)
            * self.factor
        )


class Inversion(Shape):
    def __init__(self, shape: Shape):
        super().__init__()
        self.shape = shape

    def astuple(self):
        return (self.shape,)

    def __hash__(self):
        return hash(self.astuple())

    def __eq__(self, other):
        if not isinstance(other, Inversion):
            return False
        return self.astuple() == other.astuple()

    def __repr__(self):
        return f"Inversion(\n{indent_shape(self.shape)}\n)"

    def distance(self, x: Num, y: Num) -> Expr:
        return -self.shape.distance(x, y)


class Dilate(Shape):
    def __init__(self, offset: Num, shape: Shape):
        super().__init__()
        self.offset = as_expr(offset)
        self.shape = shape

    def astuple(self):
        return self.offset, self.shape

    def __hash__(self):
        return hash(self.astuple())

    def __eq__(self, other):
        if not isinstance(other, Dilate):
            return False
        return self.astuple() == other.astuple()

    def __repr__(self):
        return f"Dilate(\n  {self.offset},\n{indent_shape(self.shape)}\n)"

    def distance(self, x: Num, y: Num) -> Expr:
        return self.shape.distance(x, y) - self.offset


class Shell(Shape):
    def __init__(self, thickness: Num, shape: Shape):
        super().__init__()
        self.shape = shape
        self.thickness = as_expr(thickness)

    def astuple(self):
        return self.thickness, self.shape

    def __hash__(self):
        return hash(self.astuple())

    def __eq__(self, other):
        if not isinstance(other, Shell):
            return False
        return self.astuple() == other.astuple()

    def __repr__(self):
        return f"Shell(\n  {self.thickness},\n{indent_shape(self.shape)}\n)"

    def distance(self, x: Num, y: Num) -> Expr:
        return self.shape.distance(x, y).abs() - self.thickness


class Morph(Shape):
    def __init__(self, t: Num, shape_1: Shape, shape_2: Shape):
        super().__init__()
        self.shape_1 = shape_1
        self.shape_2 = shape_2
        self.t = as_expr(t)

    def astuple(self):
        return self.t, self.shape_1, self.shape_2

    def __hash__(self):
        return hash(self.astuple())

    def __eq__(self, other):
        if not isinstance(other, Morph):
            return False
        return self.astuple() == other.astuple()

    def __repr__(self):
        return f"Morph(\n  {self.t},\n{indent_shape(self.shape_1)}, \n{indent_shape(self.shape_2)}\n)"

    def distance(self, x: Num, y: Num) -> Expr:
        return (1 - self.t) * self.shape_1.distance(
            x, y
        ) + self.t * self.shape_2.distance(x, y)


class Taper(Shape):
    def __init__(self, height: Num, factor: Num, shape: Shape):
        super().__init__()
        self.height = as_expr(height)
        self.factor = as_expr(factor)
        self.shape = shape

    def astuple(self):
        return self.height, self.factor, self.shape

    def __hash__(self):
        return hash(self.astuple())

    def __eq__(self, other):
        if not isinstance(other, Taper):
            return False
        return self.astuple() == other.astuple()

    def __repr__(self):
        return (
            f"Taper(\n  {self.height}, {self.factor}, \n{indent_shape(self.shape)}\n)"
        )

    def distance(self, x: Num, y: Num) -> Expr:
        s = self.height / (self.factor * y + (self.height - y))
        return self.shape.distance(as_expr(x) * s, y)
