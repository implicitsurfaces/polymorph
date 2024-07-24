from timeit import default_timer as timer

from polymorph_num.ops import observation
from polymorph_num.unit import Unit

x = observation("x")
unit = Unit(["x"]).register("2x", x * 2).compile()

start = timer()
unit = unit.minimize()
print(timer() - start)

start = timer()
unit = unit.minimize()
print(timer() - start)

start = timer()
unit = unit.minimize()
print(timer() - start)

unit = unit.observe({"x": 1.0})
assert unit.evaluate("2x") == 2.0
