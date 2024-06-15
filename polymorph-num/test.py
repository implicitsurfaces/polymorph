import ops
import jax_eval
import point

def circle_sdf(radius, center, point):
    dx = center.x - point.x
    dy = center.y - point.y
    dist = ops.sqrt(dx*dx + dy*dy)
    return dist - radius

r = ops.var(0)
c = point.Point(0,0)
d1 = circle_sdf(r, c, point.Point(1, 1))

v = ops.vec([1., 2.])
d2 = circle_sdf(r, c, point.Point(v, v))

d3 = ops.sum(d2)

print(d2)
print(jax_eval.eval(d1, [1]))
print(jax_eval.eval(d1, [2]))
print(jax_eval.eval(d2, [1]))
print(jax_eval.eval(d3, [1]))
