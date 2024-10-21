import jax.numpy as jnp
import matplotlib.pyplot as plt
from polymorph_num.expr import ZERO
from polymorph_num.unit import CompiledUnit, Unit
from polymorph_s2df.devutils import grids

from .eval import constraint_loss, sketch_shape
from .eval import debug_inner_node as debug_inner_node
from .nodes import (
    Angle,
    AngleLiteral,
    AngleParam,
    ArcBulge,
    ArcTangentEnd,
    ArcTangentStart,
    ArcWithSmoothEnd,
    ArcWithSmoothStart,
    Biarc,
    BiarcWithSmoothEnd,
    BiarcWithSmoothExtremities,
    BiarcWithSmoothStart,
    CartesianVector,
    ConstraintOnAngle,
    ConstraintOnDistance,
    ConstraintOnPointCoincidence,
    ConstraintOnShapeBoundary,
    Distance,
    DistanceLiteral,
    DistanceParam,
    LeftBiarc,
    Line,
    Path,
    PathClose,
    PathEdge,
    PathStart,
    Point,
    PolarVector,
    Q1Angle,
    RealParam,
    RealValue,
    RightBiarc,
    Shape,
    ShapeCircle,
    ShapeDifference,
    ShapeIntersection,
    ShapeMorph,
    ShapeRotation,
    ShapeScale,
    ShapeShell,
    ShapeTranslation,
    ShapeUnion,
    Vector,
)


def real_param() -> RealValue:
    return RealParam()


def distance_param() -> Distance:
    return DistanceParam()


def angle_param() -> Angle:
    return AngleParam()


def vector_param() -> Vector:
    return CartesianVector(RealParam(), RealParam())


def polar_vector_param() -> Vector:
    return PolarVector(AngleParam(), DistanceParam())


def point_param() -> Point:
    return vector_param().from_origin()


def as_distance(distance: float | Distance) -> Distance:
    if isinstance(distance, Distance):
        return distance
    return DistanceLiteral(distance)


def as_angle(angle: float | Angle) -> Angle:
    if isinstance(angle, Angle):
        return angle
    return AngleLiteral(angle)


def as_vector(vector: tuple[float, float] | Vector) -> Vector:
    if isinstance(vector, Vector):
        return vector
    x, y = vector
    return CartesianVector(x, y)


def as_point(point: tuple[float, float] | Point) -> Point:
    if isinstance(point, Point):
        return point
    x, y = point
    return CartesianVector(x, y).from_origin()


def as_polar_vector(vector: tuple[float | Angle, float | Distance] | Vector) -> Vector:
    if isinstance(vector, Vector):
        return vector
    angle, radius = vector
    return PolarVector(as_angle(angle), as_distance(radius))


class ShapeEditor:
    def __init__(self, shape: Shape):
        self.shape = shape

    def translate(self, vector: tuple[float, float] | Vector):
        self.shape = ShapeTranslation(self.shape, as_vector(vector))
        return self

    def rotate(self, angle: float | Angle):
        self.shape = ShapeRotation(self.shape, as_angle(angle))
        return self

    def union(self, other: "ShapeEditor"):
        self.shape = ShapeUnion(self.shape, other.shape)
        return self

    def intersect(self, other: "ShapeEditor"):
        self.shape = ShapeIntersection(self.shape, other.shape)
        return self

    def diff(self, other: "ShapeEditor"):
        self.shape = ShapeDifference(self.shape, other.shape)
        return self

    def shell(self, thickness: Distance | float | int):
        self.shape = ShapeShell(self.shape, as_distance(thickness))
        return self

    def scale(self, factor: Distance | float | int):
        self.shape = ShapeScale(self.shape, as_distance(factor))
        return self

    def morph(self, other: "ShapeEditor", t: Distance | float | int):
        self.shape = ShapeMorph(self.shape, other.shape, as_distance(t))
        return self


class PointCreator:
    def __init__(self, current_point: Point, done):
        self.current_point = current_point
        self._done = done

    def _return_point(self, point: Point, corner_radius=None):
        return self._done(point, corner_radius)

    def go_to(self, x, y, corner_radius=None):
        p = CartesianVector(x, y)
        return self._return_point(p.from_origin(), corner_radius)

    def go_to_polar(self, angle, radius, corner_radius=None):
        p = PolarVector(as_angle(angle), as_distance(radius))
        return self._return_point(p.from_origin(), corner_radius)

    def go_to_point(self, point, corner_radius=None):
        return self._return_point(point, corner_radius)

    def move_by(self, x, y, corner_radius=None):
        p = self.current_point + CartesianVector(x, y)
        return self._return_point(p, corner_radius)

    def move_by_polar(self, angle, radius, corner_radius=None):
        p = self.current_point + PolarVector(as_angle(angle), as_distance(radius))
        return self._return_point(p, corner_radius)

    def horizontal_move_by(self, x, corner_radius=None):
        return self.move_by(x, 0, corner_radius)

    def horizontal_go_to(self, x, corner_radius=None):
        return self.go_to(x, 0, corner_radius)

    def vertical_move_by(self, y, corner_radius=None):
        return self.move_by(0, y, corner_radius)

    def vertical_go_to(self, y, corner_radius=None):
        return self.go_to(0, y, corner_radius)

    def close(self) -> ShapeEditor:
        return self._done(None)


class EdgeMaker:
    def __init__(self, current_point: Point, done):
        self._done = done
        self.current_point = current_point

    def line(self):
        return self._done(Line())

    def arc(self, bulge):
        return self._done(ArcBulge(bulge))

    def arc_tangent_start(self, angle):
        return self._done(ArcTangentStart(as_angle(angle)))

    def arc_tangent_end(self, angle):
        return self._done(ArcTangentEnd(as_angle(angle)))

    def arc_smooth_start(self):
        return self._done(ArcWithSmoothStart())

    def arc_smooth_end(self):
        return self._done(ArcWithSmoothEnd())

    def biarc(self, start_angle, end_angle):
        return self._done(
            Biarc(as_angle(start_angle), as_angle(end_angle), as_angle(0))
        )

    def biarc_with_param(self, start_angle, end_angle, p=0):
        return self._done(
            Biarc(as_angle(start_angle), as_angle(end_angle), as_angle(p))
        )

    def biarc_smooth_start(self, end_angle):
        return self._done(BiarcWithSmoothStart(as_angle(end_angle), as_angle(0)))

    def biarc_smooth_end(self, start_angle):
        return self._done(BiarcWithSmoothEnd(as_angle(start_angle), as_angle(0)))

    def biarc_smooth_extremities(self):
        return self._done(BiarcWithSmoothExtremities(as_angle(0)))

    def left_biarc(self, start_angle, end_angle):
        return self._done(
            LeftBiarc(Q1Angle(as_angle(start_angle)), Q1Angle(as_angle(end_angle)))
        )

    def right_biarc(self, start_angle, end_angle):
        return self._done(
            RightBiarc(Q1Angle(as_angle(start_angle)), Q1Angle(as_angle(end_angle)))
        )


def draw_circle(radius: float, origin: tuple[float, float] = (0, 0)):
    return ShapeEditor(ShapeCircle(as_distance(radius), as_point(origin)))


def draw(origin: tuple[float, float] = (0, 0), corner_radius=None):
    current_point = as_point(origin)
    path: Path = PathStart(
        current_point, as_distance(corner_radius) if corner_radius is not None else None
    )

    def line_done(line):
        def point_done(point, corner_radius=None):
            nonlocal path
            if point is None:
                return ShapeEditor(PathClose(path, line))

            nonlocal current_point
            current_point = point
            path = PathEdge(
                path,
                line,
                point,
                as_distance(corner_radius) if corner_radius is not None else None,
            )
            return EdgeMaker(current_point, line_done)

        nonlocal current_point
        return PointCreator(current_point, point_done)

    return EdgeMaker(current_point, line_done)


class LossMaker:
    def __init__(self):
        self.constraints = []

    def fit_distance(self, distance, target, tol=1e-3):
        self.constraints.append(
            ConstraintOnDistance(as_distance(distance), target, tol)
        )
        return self

    def fit_angle(self, angle, target, tol=1e-3):
        self.constraints.append(ConstraintOnAngle(as_angle(angle), target, tol))
        return self

    def fit_point(self, point, target, tol=1e-3):
        self.constraints.append(
            ConstraintOnPointCoincidence(as_point(point), as_point(target), tol)
        )
        return self

    def fit_point_on_boundary(self, shape, point, tol=1e-3):
        self.constraints.append(ConstraintOnShapeBoundary(shape, as_point(point), tol))
        return self

    def create_sketch(self):
        loss = sum([constraint_loss(c) for c in self.constraints], ZERO)
        return Sketch(loss)


def loss():
    return LossMaker()


class Sketch:
    def __init__(self, loss=None):
        self.shapes = []
        self.loss = loss
        self._shape_names = {}
        self._compiled = None

        self._samples = 500
        self._bounds = (-3, 3)

        self._grids = grids(self._samples, self._bounds)

    def add_shape(self, shape: ShapeEditor, name=None):
        self.shapes.append(shape.shape)
        if name:
            self._shape_names[name] = len(self.shapes) - 1
        self._compiles = None
        return self

    def add_loss(self, loss):
        self.loss = loss
        return self

    def _compile(self):
        unit = Unit()
        if self.loss:
            unit.registerLoss(self.loss)

        grid_x, grid_y = self._grids[0]
        for i, shape in enumerate(self.shapes):
            unit.register(f"shape_{i}", sketch_shape(shape).distance(grid_x, grid_y))

        c = unit.compile()
        c = c.minimize()

        self._compiled = c
        return c

    @property
    def _unit(self) -> CompiledUnit:
        if self._compiled is None:
            return self._compile()
        return self._compiled

    def debug(self, node):
        return self._unit.run(debug_inner_node(node))

    def _plot_dist(self, values):
        _, ax2 = plt.subplots(layout="constrained")
        X, Y = self._grids[1]
        bounds = self._bounds

        values = values.reshape(self._samples, self._samples)

        levels = jnp.linspace(-5, 5, 41)

        ax2.axis("equal")
        ax2.contourf(
            X,
            Y,
            values,
            levels=levels,
            cmap="PRGn",
            origin="lower",
            extent=[bounds[0], bounds[1], bounds[0], bounds[1]],
        )
        ax2.contour(
            X,
            Y,
            values,
            levels=levels,
            colors="k",
            origin="lower",
            extent=[bounds[0], bounds[1], bounds[0], bounds[1]],
        )

    def _plot_is_inside(self, values):
        values = values.reshape(self._samples, self._samples)
        bounds = self._bounds

        plt.imshow(
            values > 0,
            cmap="gray",
            origin="lower",
            extent=(bounds[0], bounds[1], bounds[0], bounds[1]),
        )

    def _get_values(self, index):
        if isinstance(index, ShapeEditor):
            return self._unit.run(sketch_shape(index.shape).distance(*self._grids[0]))

        if index is None:
            index = 0
        if isinstance(index, str):
            index = self._shape_names[index]

        name = f"shape_{index}"
        return self._unit.evaluate(name)

    def plot(self, index=None):
        values = self._get_values(index)
        self._plot_dist(values)

    def render(self, index=None):
        values = self._get_values(index)
        self._plot_is_inside(values)
