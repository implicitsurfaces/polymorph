import ops
import optimizer
import point
import loss

def circle_sdf(radius, center, point):
    dx = center.x - point.x
    dy = center.y - point.y
    dist = ops.sqrt(dx*dx + dy*dy)
    return dist - radius

r = ops.param()
c = point.Point(0,0)
d = circle_sdf(r, c, point.Point(1, 1))
l = loss.Loss(d*d)
l.register_output(r)
opt = optimizer.Optimizer(l)
soln = opt.optimize({})
print(soln.eval(r))
#solution = optimizer.minimize(loss)
#print(solution.eval(r))

l.register_output(d)

v = ops.vec([1., 3.])
d2 = circle_sdf(r, c, point.Point(v, v))
#print(solution.eval(ops.sum(d2)))