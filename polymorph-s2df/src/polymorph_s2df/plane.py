from polymorph_num.expr import ZERO, Num
from polymorph_num.vec3 import ORIGIN, X_AXIS, Y_AXIS, Z_AXIS, ValVec3, Vec3, as_vec3


class Plane:
    origin: Vec3
    zAxis: Vec3
    xAxis: Vec3

    def __init__(self, zAxis=Z_AXIS, origin=ORIGIN, xAxis=X_AXIS):
        self.origin = origin
        self.zAxis = zAxis.normalize()
        self.xAxis = xAxis.normalize()

    @property
    def yAxis(self):
        return self.zAxis.cross(self.xAxis)

    def translateTo(self, new_origin: ValVec3):
        return Plane(self.zAxis, as_vec3(new_origin), self.xAxis)

    def translate(self, x: Num = ZERO, y: Num = ZERO, z: Num = ZERO):
        return Plane(self.zAxis, self.origin + Vec3(x, y, z), self.xAxis)

    def pivot(self, angle: Num, axis: ValVec3):
        ax = as_vec3(axis)
        return Plane(
            self.zAxis.rotate(angle, ax),
            self.origin,
            self.xAxis.rotate(angle, ax),
        )

    def rotate2DAxes(self, angle: Num):
        return Plane(self.zAxis, self.origin, self.xAxis.rotate(angle, self.zAxis))

    def local_coordinates(self, point: ValVec3):
        p = as_vec3(point) - self.origin
        return Vec3(p.dot(self.xAxis), p.dot(self.yAxis), p.dot(self.zAxis))

    def global_coordinates(self, point: ValVec3):
        p = as_vec3(point)
        return (
            self.origin
            + self.xAxis.scale(p.x)
            + self.yAxis.scale(p.y)
            + self.zAxis.scale(p.z)
        )


XY_PLANE = Plane(Z_AXIS, ORIGIN, X_AXIS)
XZ_PLANE = Plane(Y_AXIS, ORIGIN, X_AXIS)
YZ_PLANE = Plane(X_AXIS, ORIGIN, Y_AXIS)
