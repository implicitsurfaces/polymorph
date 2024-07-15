import math

import polymorph_s2df as s2df
import pytest
from polymorph_num.ops import grid_gen, observation, param
from polymorph_num.unit import Unit
from polymorph_num.vec import Vec2


def circle_sdf(radius, center, point):
    dx = center.x - point.x
    dy = center.y - point.y
    dist = (dx * dx + dy * dy).sqrt()
    return dist - radius


def cumulative_sum(n):
    i = observation("x")
    sum = i
    for _ in range(n):
        i = i + 1
        sum = sum + i
    return sum


def cumulative_n(n):
    unit = (
        Unit(["x"])
        .register("sum", cumulative_sum(n))
        .compile()
        .minimize()
        .observe({"x": 0.0})
    )
    return unit.evaluate("sum")


def test_minimal():
    x = observation("x")
    unit = Unit(["x"]).register("2x", x * 2).compile().minimize().observe({"x": 1.0})
    assert unit.evaluate("2x") == 2.0


def test_cumulative(benchmark):
    result = benchmark(cumulative_n, 100)
    assert result == 5050


def test_optimization_with_units():
    # For circle at (0, 0), find the radius by minimizing the distance to a
    # point determined by two observations.
    r = param()
    c = Vec2(0, 0)
    obs_pt = Vec2(observation("x"), observation("y"))
    d = circle_sdf(r, c, obs_pt)
    q = obs_pt.x * r
    unit = (
        Unit(["x", "y"])
        .registerLoss(d * d)
        .register("radius", r)
        .register("q", q)
        .compile()
        .observe({"x": 1.0, "y": 1.0})
        .minimize()
    )
    assert unit.evaluate("radius") == pytest.approx(math.sqrt(2.0))
    assert unit.evaluate("q") == pytest.approx(math.sqrt(2.0))

    unit = unit.observe({"x": 2.0, "y": 0.0}).minimize()
    assert unit.evaluate("radius") == 2.0
    assert unit.evaluate("q") == 4.0


def test_unknown_param():
    with pytest.raises(ValueError) as e:
        Unit().register("x", param()).compile()
    assert "params not found in loss" in str(e.value)


def test_unknown_obs():
    with pytest.raises(ValueError) as e:
        Unit(["x"]).register("x", observation("y")).compile()
    assert "Observation 'y' not found" in str(e.value)


def make_polygon(points):
    segments = [
        s2df.LineSegment(
            points[i],
            points[(i + 1) % len(points)],
        )
        for i in range(len(points))
    ]
    return s2df.ClosedPath(segments)


def test_is_inside_polygon(benchmark):
    def compile():
        sdf = make_polygon(
            [
                Vec2(-100, -100),
                Vec2(100, -100),
                Vec2(100, 100),
                Vec2(0, 0),
                Vec2(-100, 100),
            ]
        )

        grid = grid_gen(300, 500)

        unit = Unit()
        unit.register("is_inside", sdf.is_inside(*grid))
        return unit.compile()

    benchmark(compile)
