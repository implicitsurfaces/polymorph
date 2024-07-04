import math

import pytest
from polymorph_num import loss, ops, optimizer, vec


def circle_sdf(radius, center, point):
    dx = center.x - point.x
    dy = center.y - point.y
    dist = (dx * dx + dy * dy).sqrt()
    return dist - radius


def test_basic_optimization():
    # For circle at (0, 0), find the radius by minimizing the distance to a
    # point determined by two observations.
    r = ops.param()
    c = vec.Vec2(0, 0)
    obs_pt = vec.Vec2(ops.observation("x"), ops.observation("y"))
    d = circle_sdf(r, c, obs_pt)
    q = obs_pt.x * r
    l = loss.Loss(d * d)
    l.register_output(r)
    l.register_output(q)
    opt = optimizer.Optimizer(l)
    soln = opt.optimize({"x": 1.0, "y": 1.0})
    assert soln.eval(r).item() == pytest.approx(math.sqrt(2.0))
    assert soln.eval(q).item() == pytest.approx(math.sqrt(2.0))

    soln2 = opt.optimize({"x": 2.0, "y": 0.0})
    assert soln2.eval(r).item() == 2.0
    assert soln2.eval(q).item() == 4.0
