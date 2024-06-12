import jax.numpy as jnp
from jax import grad

from polymorph_s2df import *


import optimistix
from timeit import default_timer as timer


def optimize_params(cost, params, scene):
    solver = optimistix.BFGS(rtol=1e-5, atol=1e-6)
    start = timer()
    solution = optimistix.minimise(cost, solver, params, scene, throw=False)
    elapsed = timer() - start
    print(
        "{0} steps in {1:.3f} seconds".format(solution.stats.get("num_steps"), elapsed)
    )
    return solution.value


def cost(params, shape):
    target_distance = 0.5
    cost_distance = (shape.distance(params[jnp.newaxis, 0:2]) - target_distance) ** 2
    return cost_distance[0]


def async_solver(pool):
    value = None
    currently_processing = False

    def solver(params, scene):
        nonlocal value, currently_processing

        if currently_processing and currently_processing.ready():
            value = currently_processing.get()
            currently_processing = False

        if not currently_processing:
            currently_processing = pool.apply_async(
                optimize_params, (cost, params, scene)
            )

        g = grad(cost)(params, scene)
        print(f"grad {g}")

        return value if value is not None else params

    return solver


def sync_solver(params, scene):
    return optimize_params(cost, params, scene)
