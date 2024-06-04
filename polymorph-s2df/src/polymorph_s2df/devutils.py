import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from .operations import Shape


def render_distance(shape, bounds=(-3, 3), n=500):
    x = jnp.linspace(bounds[0], bounds[1], n)
    X, Y = jnp.meshgrid(x, x)

    grid_points = jnp.column_stack((X.flatten(), Y.flatten()))
    sdf = shape.distance

    values = sdf(grid_points).reshape(n, n)

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


def render(shape, bounds=(-3, 3), n=500):
    x = jnp.linspace(bounds[0], bounds[1], n)
    X, Y = jnp.meshgrid(x, x)

    grid_points = jnp.column_stack((X.flatten(), Y.flatten()))

    def vec_is_inside(x):
        return 1 - shape.is_inside(x)

    plt.imshow(
        vec_is_inside(grid_points).reshape(n, n),
        cmap="gray",
        origin="lower",
        extent=[bounds[0], bounds[1], bounds[0], bounds[1]],
    )


class S(Shape):
    """A shape adapter for segments"""

    def __init__(self, segment):
        self.segment = segment

    def distance(self, p):
        distance, _ = jax.vmap(self.segment.distance_and_mask)(p)
        return distance
