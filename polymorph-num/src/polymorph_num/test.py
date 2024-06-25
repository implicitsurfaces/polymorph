from . import loss, ops, optimizer, point


def circle_sdf(radius, center, point):
    dx = center.x - point.x
    dy = center.y - point.y
    dist = (dx * dx + dy * dy).sqrt()
    return dist - radius


r = ops.param()
c = point.Point(0, 0)
obs_pt = point.Point(ops.observation("x"), ops.observation("y"))
d = circle_sdf(r, c, obs_pt)
q = obs_pt.x * r
l = loss.Loss(d * d)
l.register_output(r)
l.register_output(q)
opt = optimizer.Optimizer(l)
print("---")
soln = opt.optimize({"x": 1.0, "y": 1.0})
soln2 = opt.optimize({"x": 2.0, "y": 1.0})
print(soln.eval(r))
print(soln.eval(q))
print(soln2.eval(r))
print(soln2.eval(q))
