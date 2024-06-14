import ops
import float_eval
import point

def circle_sdf(radius, center, point):
    dx = center.x - point.x
    dy = center.y - point.y
    dist = ops.sqrt(dx*dx + dy*dy)
    return dist - radius

r = ops.var(0)
c = point.Point(0,0)
d1 = circle_sdf(r, c, point.Point(1, 1))

print(float_eval.eval(d1, [1]))
print(float_eval.eval(d1, [2]))