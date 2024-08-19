import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import polymorph_num.expr as e
import polyscope as ps
from polymorph_num.expr import Expr
from polymorph_num.ops import grid_gen, grid_gen_3d
from polymorph_num.unit import Unit
from polymorph_num.vec import Vec2
from polymorph_num.vec3 import ORIGIN, X_AXIS, Y_AXIS, Z_AXIS

from polymorph_s2df.embed import Solid
from polymorph_s2df.plane import Plane

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


def eval_distance(shape, x, y):
    return eval_expr(shape.distance(x, y))


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


def render_distance(shape: Shape, bounds=(-3, 3), n=500):
    (grid_x, grid_y), (X, Y) = grids(n, bounds)

    values = eval_distance(shape, grid_x, grid_y).reshape(n, n)

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


def render(shape: Shape, bounds=(-3, 3), n=500):
    (grid_x, grid_y), (X, Y) = grids(n, bounds)

    values = eval_distance(shape, grid_x, grid_y).reshape(n, n)

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


def render_solid(solid: Solid, bounds=(-3, 3), n=100, show_distances=False):
    def rescale_grid(b):
        diff = bounds[1] - bounds[0]
        return (b / n + 0.5) * diff + bounds[0]

    (grid_x, grid_y, grid_z) = grid_gen_3d(n, n, n)

    dims = (n, n, n)
    bound_low = (bounds[0], bounds[0], bounds[0])
    bound_high = (bounds[1], bounds[1], bounds[1])

    distance = np_eval(
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


def np_eval(
    expr: e.Expr, params=None, param_map=None, obs_dict=None, random_key=1, memo=None
) -> np.ndarray:
    if memo is None:
        memo = {}
    if param_map is None:
        param_map = {}
    if obs_dict is None:
        obs_dict = {}

    if expr in memo:
        return memo[expr]

    result = None
    match expr:
        case e.Scalar(value):
            result = np.array(value)

        case e.Arr(value):
            result = np.array(value)

        case e.GridX(width, height):
            half_width = width / 2
            half_height = height / 2

            yy, xx = np.mgrid[-half_height:half_height, -half_width:half_width]
            result = xx.ravel()

        case e.GridY(width, height):
            half_width = width / 2
            half_height = height / 2

            yy, xx = np.mgrid[-half_height:half_height, -half_width:half_width]
            result = yy.ravel()

        case e.GridX3d(width, height, depth):
            half_width = width / 2
            half_height = height / 2
            half_depth = depth / 2

            xx, yy, zz = np.mgrid[
                -half_width:half_width, -half_height:half_height, -half_depth:half_depth
            ]
            result = xx.ravel()

        case e.GridY3d(width, height, depth):
            half_width = width / 2
            half_height = height / 2
            half_depth = depth / 2

            xx, yy, zz = np.mgrid[
                -half_width:half_width, -half_height:half_height, -half_depth:half_depth
            ]
            result = yy.ravel()

        case e.GridZ3d(width, height, depth):
            half_width = width / 2
            half_height = height / 2
            half_depth = depth / 2

            xx, yy, zz = np.mgrid[
                -half_width:half_width, -half_height:half_height, -half_depth:half_depth
            ]
            result = zz.ravel()

        case e.Random(dim, low, high):
            min_ = np_eval(low, params, param_map, obs_dict, random_key, memo)
            max_ = np_eval(high, params, param_map, obs_dict, random_key, memo)

            result = jax.random.uniform(
                next(random_key), shape=(dim,), minval=min_, maxval=max_
            )

        case e.Broadcast(orig, _dim):
            result = np_eval(orig, params, param_map, obs_dict, random_key, memo)

        case e.Unary(orig, op, constants):
            o = np_eval(orig, params, param_map, obs_dict, random_key, memo)
            match op:
                case e.UnOp.Sqrt:
                    result = np.sqrt(o)
                case e.UnOp.Sigmoid:
                    result = jax.nn.sigmoid(o)
                case e.UnOp.SmoothAbs:
                    result = o * np.tanh(10.0 * o)
                case e.UnOp.Abs:
                    result = np.abs(o)
                case e.UnOp.SoftPlus:
                    result = jax.nn.softplus(50 * o) / 50
                case e.UnOp.Log:
                    result = np.log(o)
                case e.UnOp.Exp:
                    result = np.exp(o)
                case e.UnOp.Cos:
                    result = np.cos(o)
                case e.UnOp.Sin:
                    result = np.sin(o)
                case e.UnOp.Sign:
                    result = np.sign(o)
                case e.UnOp.Tanh:
                    result = np.tanh(o)
                case e.UnOp.ArcTan:
                    result = np.atan(o)
                case e.UnOp.Boxcar:
                    min, max = constants
                    result = np.where((o >= min) & (o <= max), 1.0, 0.0)

        case e.Binary(left, right, op):
            l = np_eval(left, params, param_map, obs_dict, random_key, memo)
            r = np_eval(right, params, param_map, obs_dict, random_key, memo)
            match op:
                case e.BinOp.Add:
                    result = l + r
                case e.BinOp.Sub:
                    result = l - r
                case e.BinOp.Mul:
                    result = l * r
                case e.BinOp.Div:
                    result = l / r
                case e.BinOp.Exp:
                    result = l**r
                case e.BinOp.Min:
                    result = np.minimum(l, r)
                case e.BinOp.Max:
                    result = np.maximum(l, r)
                case e.BinOp.ArcTan2:
                    result = np.arctan2(l, r)
                case e.BinOp.Mod:
                    result = np.mod(l, r)

        case e.ComparisonIf(a, b, condition_true, condition_false, op):
            a_ = np_eval(a, params, param_map, obs_dict, random_key, memo=memo)
            b_ = np_eval(b, params, param_map, obs_dict, random_key, memo=memo)
            if op == e.ComparisonOp.Gt:
                result = np.where(
                    a_ > b_,
                    np_eval(
                        condition_true,
                        params,
                        param_map,
                        obs_dict,
                        random_key,
                        memo,
                    ),
                    np_eval(
                        condition_false,
                        params,
                        param_map,
                        obs_dict,
                        random_key,
                        memo,
                    ),
                )
            elif op == e.ComparisonOp.Ge:
                result = np.where(
                    a_ >= b_,
                    np_eval(
                        condition_true,
                        params,
                        param_map,
                        obs_dict,
                        random_key,
                        memo,
                    ),
                    np_eval(
                        condition_false,
                        params,
                        param_map,
                        obs_dict,
                        random_key,
                        memo,
                    ),
                )
            elif op == e.ComparisonOp.Eq:
                result = np.where(
                    a_ == b_,
                    np_eval(
                        condition_true,
                        params,
                        param_map,
                        obs_dict,
                        random_key,
                        memo,
                    ),
                    np_eval(
                        condition_false,
                        params,
                        param_map,
                        obs_dict,
                        random_key,
                        memo,
                    ),
                )

        case e.Param():
            result = param_map.get(expr, params)

        case e.Observation(name):
            result = obs_dict[name]

        case e.Sum(orig):
            result = np.sum(
                np_eval(orig, params, param_map, obs_dict, random_key, memo)
            )

        case e.Debug(tag, orig):
            result = np_eval(orig, params, param_map, obs_dict, random_key, memo)
            print(tag, result)

        case _:
            raise ValueError(f"Unknown expression: {expr}")
    assert result is not None
    memo[expr] = result
    return result
