import ops
import jax_eval
import point
import params

def circle_sdf(radius, center, point):
    dx = center.x - point.x
    dy = center.y - point.y
    dist = ops.sqrt(dx*dx + dy*dy)
    return dist - radius

r = ops.param() * 5
c = point.Point(0,0)
d = circle_sdf(r, c, point.Point(1, 1))
loss = d*d
solution = jax_eval.minimize(loss)
print(solution.eval(r))
