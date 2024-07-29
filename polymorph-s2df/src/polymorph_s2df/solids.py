from polymorph_num.expr import Num, as_expr
from polymorph_num.vec3 import Vec3

from polymorph_s2df.solid_operations import Solid


class Sphere(Solid):
    def __init__(self, radius: Num):
        self.radius = as_expr(radius)

    def distance(self, x: Num, y: Num, z: Num):
        return Vec3(x, y, z).norm() - self.radius
