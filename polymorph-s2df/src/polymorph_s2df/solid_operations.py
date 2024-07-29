from polymorph_num import ops
from polymorph_num.expr import Expr, Num, as_expr
from polymorph_num.vec3 import Z_AXIS, ValVec3, as_vec3


class Solid:
    def distance(self, x: Num, y: Num, z: Num) -> Expr:
        raise NotImplementedError()

    def is_inside(self, x: Num, y: Num, z: Num, scale=100) -> Expr:
        return 1 - (self.distance(x, y, z) * scale).sigmoid()

    def is_on_boundary(self, x: Num, y: Num, z: Num, dist=2.0) -> Expr:
        return self.distance(x, y, z).boxcar(-dist / 2.0, dist / 2.0)

    def area(self, sample_xs: Num, sample_ys: Num, sample_zs: Num, sample_area: Num):
        return ops.mean(self.is_inside(sample_xs, sample_ys, sample_zs)) * sample_area

    def astuple(self):
        raise NotImplementedError

    def bounding_box(self):
        raise NotImplementedError

    def __hash__(self):
        return hash(self.astuple())

    def union(self, other: "Solid"):
        return Union(self, other)

    def intersect(self, other: "Solid"):
        return Intersection(self, other)

    def substract(self, other: "Solid"):
        return Substraction(self, other)

    def smooth_union(self, k: Num, other: "Solid"):
        return SmoothUnion(k, self, other)

    def smooth_intersect(self, k: Num, other: "Solid"):
        return SmoothIntersection(k, self, other)

    def smooth_substract(self, k, other: "Solid"):
        return SmoothSubstraction(k, self, other)

    def shell(self, thickness: Num):
        return Shell(thickness, self)

    def translate(self, offset: ValVec3):
        return Translation(offset, self)

    def rotate(self, angle: Num):
        return Rotation(angle, self)

    def scale(self, factor: Num):
        return Scale(factor, self)

    def invert(self):
        return Inversion(self)

    def dilate(self, offset: Num):
        return Dilate(offset, self)

    def morph(self, t: Num, other: "Solid"):
        return Morph(t, self, other)


class Translation(Solid):
    def __init__(self, offset: ValVec3, shape: Solid):
        super().__init__()
        self.offset = as_vec3(offset)
        self.shape = shape

    def astuple(self):
        return self.offset, self.shape

    def __hash__(self):
        return hash(self.astuple())

    def __eq__(self, other):
        if not isinstance(other, Translation):
            return False
        return self.astuple() == other.astuple()

    def distance(self, x: Num, y: Num, z: Num) -> Expr:
        return self.shape.distance(
            x - self.offset.x, y - self.offset.y, z - self.offset.z
        )


class Rotation(Solid):
    def __init__(self, angle: Num, shape: Solid):
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

    def distance(self, x: Num, y: Num, z: Num) -> Expr:
        vec = as_vec3((x, y, z))
        v2 = vec.rotate(self.angle, Z_AXIS)

        return self.shape.distance(v2.x, v2.y, v2.z)


class Intersection(Solid):
    def __init__(self, shape_1: Solid, shape_2: Solid):
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

    def distance(self, x: Num, y: Num, z: Num) -> Expr:
        return ops.max(self.shape_1.distance(x, y, z), self.shape_2.distance(x, y, z))


class SmoothIntersection(Solid):
    def __init__(self, k: Num, shape_1: Solid, shape_2: Solid):
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

    def distance(self, x: Num, y: Num, z: Num) -> Expr:
        val1 = self.shape_1.distance(x, y, z)
        val2 = self.shape_2.distance(x, y, z)

        h = ops.max(self.k - (val1 - val2).abs(), 0.0) / self.k
        return ops.max(val1, val2) + h * h * self.k * 1.0 / 4.0


class Union(Solid):
    def __init__(self, shape_1: Solid, shape_2: Solid):
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

    def distance(self, x: Num, y: Num, z: Num) -> Expr:
        return ops.min(self.shape_1.distance(x, y, z), self.shape_2.distance(x, y, z))


class SmoothUnion(Solid):
    def __init__(self, k, shape_1: Solid, shape_2: Solid):
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

    def distance(self, x: Num, y: Num, z: Num) -> Expr:
        val1 = self.shape_1.distance(x, y, z)
        val2 = self.shape_2.distance(x, y, z)

        h = ops.max(self.k - (val1 - val2).abs(), 0.0) / self.k
        return ops.min(val1, val2) - h * h * self.k * 1.0 / 4.0


class Substraction(Solid):
    def __init__(self, shape_1: Solid, shape_2: Solid):
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

    def distance(self, x: Num, y: Num, z: Num) -> Expr:
        return ops.max(self.shape_1.distance(x, y, z), -self.shape_2.distance(x, y, z))


class SmoothSubstraction(Solid):
    def __init__(self, k: Num, shape_1: Solid, shape_2: Solid):
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

    def distance(self, x: Num, y: Num, z: Num) -> Expr:
        val1 = self.shape_1.distance(x, y, z)
        val2 = self.shape_2.distance(x, y, z)

        h = ops.max(self.k - (val1 + val2).abs(), 0.0) / self.k
        return ops.max(val1, -val2) + h * h * self.k * (1.0 / 4.0)


class Scale(Solid):
    def __init__(self, factor: Num, shape: Solid):
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

    def distance(self, x: Num, y: Num, z: Num) -> Expr:
        return (
            self.shape.distance(x / self.factor, y / self.factor, z / self.factor)
            * self.factor
        )


class Inversion(Solid):
    def __init__(self, shape: Solid):
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

    def distance(self, x: Num, y: Num, z: Num) -> Expr:
        return -self.shape.distance(x, y, z)


class Dilate(Solid):
    def __init__(self, offset: Num, shape: Solid):
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

    def distance(self, x: Num, y: Num, z: Num) -> Expr:
        return self.shape.distance(x, y, z) - self.offset


class Shell(Solid):
    def __init__(self, thickness: Num, shape: Solid):
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

    def distance(self, x: Num, y: Num, z: Num) -> Expr:
        return self.shape.distance(x, y, z).abs() - self.thickness


class Morph(Solid):
    def __init__(self, t: Num, shape_1: Solid, shape_2: Solid):
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

    def distance(self, x: Num, y: Num, z: Num) -> Expr:
        return (1 - self.t) * self.shape_1.distance(
            x, y, z
        ) + self.t * self.shape_2.distance(x, y, z)
