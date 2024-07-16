from polymorph_num import ops
from polymorph_num.vec import Vec2


def area(shape, size):
    grid = ops.grid_gen(*size)
    return ops.mean(shape.is_inside(*grid)) * size[0] * size[1]


def area_monte_carlo(shape, size, n=1000):
    x = ops.random_uniform(-size[0] / 2, size[0] / 2, n)
    y = ops.random_uniform(-size[1] / 2, size[1] / 2, n)
    return ops.mean(shape.is_inside(x, y)) * size[0] * size[1]


def centroid(shape, size) -> Vec2:
    grid = ops.grid_gen(*size)

    is_inside = shape.is_inside(*grid)
    total_weight = ops.sum(is_inside)

    x_mean = ops.sum(grid[0] * is_inside) / total_weight
    y_mean = ops.sum(grid[1] * is_inside) / total_weight

    return Vec2(x_mean, y_mean)


def centroid_monte_carlo(shape, size, n=1000) -> Vec2:
    x = ops.random_uniform(-size[0] / 2, size[0] / 2, n)
    y = ops.random_uniform(-size[1] / 2, size[1] / 2, n)

    is_inside = shape.is_inside(x, y)
    total_weight = ops.sum(is_inside)

    x_mean = ops.sum(x * is_inside) / total_weight
    y_mean = ops.sum(y * is_inside) / total_weight

    return Vec2(x_mean, y_mean)
