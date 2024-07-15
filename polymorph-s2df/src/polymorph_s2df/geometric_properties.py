from polymorph_num import ops
from polymorph_num.vec import Vec2


def area(shape, size):
    grid = ops.grid_gen(*size)
    return ops.mean(shape.is_inside(*grid)) * size[0] * size[1]


def centroid(shape, size) -> Vec2:
    grid = ops.grid_gen(*size)

    is_inside = shape.is_inside(*grid)

    x_mean = ops.mean(grid[0] * is_inside)
    y_mean = ops.mean(grid[1] * is_inside)

    return Vec2(x_mean, y_mean)
