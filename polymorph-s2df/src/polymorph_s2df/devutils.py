import jax.numpy as jnp
import matplotlib.pyplot as plt
import polyscope as ps
from polymorph_num.expr import Expr
from polymorph_num.ops import grid_gen, grid_gen_3d
from polymorph_num.unit import Unit
from polymorph_num.vec import Vec2
from polymorph_num.vec3 import ORIGIN, X_AXIS, Y_AXIS, Z_AXIS

from polymorph_s2df.embed import Solid
from polymorph_s2df.plane import Plane

from .numpy_eval import np_eval as np_eval
from .operations import Shape
from .paths import PathSegment

# Helpers for working for 3D axes
X_AXIS = X_AXIS
Y_AXIS = Y_AXIS
Z_AXIS = Z_AXIS
ORIGIN = ORIGIN


def p(x, y):
    return Vec2(x, y)


def eval_expr(expr: Expr):
    return Unit().register("result", expr).compile().evaluate("result")


def grids(n, bounds):
    def rescale_grid(b):
        diff = bounds[1] - bounds[0]
        return (b / n + 0.5) * diff + bounds[0]

    grid_x, grid_y = grid_gen(n, n)

    half_width = n / 2
    half_height = n / 2

    Y, X = rescale_grid(jnp.mgrid[-half_height:half_height, -half_width:half_width])

    return (
        (rescale_grid(grid_x), rescale_grid(grid_y)),
        (X, Y),
    )


def render_distance(shape: Shape, bounds=(-3, 3), n=500, eval_fn=eval_expr):
    (grid_x, grid_y), (X, Y) = grids(n, bounds)

    values = eval_fn(shape.distance(grid_x, grid_y)).reshape(n, n)

    _, ax2 = plt.subplots(layout="constrained")

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


def render(shape: Shape, bounds=(-3, 3), n=500, eval_fn=eval_expr):
    (grid_x, grid_y), (X, Y) = grids(n, bounds)

    values = eval_fn(shape.distance(grid_x, grid_y)).reshape(n, n)

    plt.imshow(
        values > 0,
        cmap="gray",
        origin="lower",
        extent=(bounds[0], bounds[1], bounds[0], bounds[1]),
    )


def compute_winding_number(segment, X, Y):
    return eval_expr(segment.winding_number(Vec2(X, Y)))


def render_winding_number(segment: PathSegment, bounds=(-3, 3), n=500):
    (grid_x, grid_y), (X, Y) = grids(n, bounds)
    values = compute_winding_number(segment, grid_x, grid_y).reshape(n, n)
    plt.imshow(
        values,
        origin="lower",
        cmap="coolwarm",
        extent=(bounds[0], bounds[1], bounds[0], bounds[1]),
    )
    plt.colorbar()


def render_solid(
    solid: Solid, bounds=(-3, 3), n=100, show_distances=False, eval_fn=eval_expr
):
    def rescale_grid(b):
        diff = bounds[1] - bounds[0]
        return (b / n + 0.5) * diff + bounds[0]

    (grid_x, grid_y, grid_z) = grid_gen_3d(n, n, n)

    dims = (n, n, n)
    bound_low = (bounds[0], bounds[0], bounds[0])
    bound_high = (bounds[1], bounds[1], bounds[1])

    distance = eval_fn(
        solid.distance(rescale_grid(grid_x), rescale_grid(grid_y), rescale_grid(grid_z))
    ).reshape(dims)

    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_x_front")

    # register the grid
    ps_grid = ps.register_volume_grid("sample grid", dims, bound_low, bound_high)

    # add a scalar function on the grid
    ps_grid.add_scalar_quantity(
        "distance",
        distance,
        defined_on="nodes",
        vminmax=(-5.0, 5.0),
        isosurface_level=0.0,
        cmap="coolwarm",
        enabled=True,
        enable_isosurface_viz=True,
        enable_gridcube_viz=show_distances,
        isolines_enabled=True,
        slice_planes_affect_isosurface=False,
    )

    if show_distances:
        ps_plane = ps.add_scene_slice_plane()
        ps_plane.set_draw_plane(False)  # render the semi-transparent gridded plane
        ps_plane.set_draw_widget(False)
        ps_plane.set_pose((0, 0, 0), (1, 0, 0))


class SolidSlice(Shape):
    def __init__(self, solid: Solid, plane: Plane):
        self.solid = solid
        self.plane = plane

    def distance(self, x, y):
        projected_point = self.plane.global_coordinates(Vec2(x, y))
        return self.solid.distance(
            projected_point.x, projected_point.y, projected_point.z
        )
