from polymorph_num import ops
from polymorph_num.vec import Vec2


def area(shape, n=100):
    bbox = shape.bounding_box()
    grid = ops.regular_grid(bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y, n)
    return (
        ops.mean(shape.is_inside(*grid))
        * (bbox.max_x - bbox.min_x)
        * (bbox.max_y - bbox.min_y)
    )


def area_monte_carlo(shape, n=1000):
    bbox = shape.bounding_box()
    x = ops.random_uniform(bbox.min_x, bbox.max_x, n)
    y = ops.random_uniform(bbox.min_y, bbox.max_y, n)
    return (
        ops.mean(shape.is_inside(x, y))
        * (bbox.max_x - bbox.min_x)
        * (bbox.max_y - bbox.min_y)
    )


def centroid(shape, size=100) -> Vec2:
    bbox = shape.bounding_box()
    grid = ops.regular_grid(bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y, size)

    is_inside = shape.is_inside(*grid)
    total_weight = ops.sum(is_inside)

    x_mean = ops.sum(grid[0] * is_inside) / total_weight
    y_mean = ops.sum(grid[1] * is_inside) / total_weight

    return Vec2(x_mean, y_mean)


def centroid_monte_carlo(shape, n=1000):
    bbox = shape.bounding_box()
    x = ops.random_uniform(bbox.min_x, bbox.max_x, n)
    y = ops.random_uniform(bbox.min_y, bbox.max_y, n)

    is_inside = shape.is_inside(x, y)
    total_weight = ops.sum(is_inside)

    x_mean = ops.sum(x * is_inside) / total_weight
    y_mean = ops.sum(y * is_inside) / total_weight

    return Vec2(x_mean, y_mean)
