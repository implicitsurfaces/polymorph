import random

from lineax._solver.misc import math
from polymorph_app.sketch import (
    Area,
    BottomHalfPlane,
    CenteredBox,
    Centroid,
    Circle,
    DistanceConstraint,
    Intersection,
    LengthValue,
    MathValue,
    Morph,
    Polygon,
    SameValueConstraint,
    Sketch,
    VerticallyAligned,
)


def load_chain(size=(500, 500)):
    sketch = Sketch()
    dist = LengthValue().lock(0)

    prev_circle = None
    radius_val = 20
    for _ in range(6):
        c = Circle()
        c.radius.lock(radius_val)
        sketch.add(c)
        if prev_circle is None:
            # Bind center of first circle to the mouse.
            c.center.bind("mouse_x", "mouse_y")
        else:
            # Make adjacent circles just touch.
            dist = LengthValue().lock(radius_val * 2)
            constraint = DistanceConstraint(prev_circle.center, c.center, dist)
            sketch.add_constraint(constraint)
        prev_circle = c
    return sketch


def empty(size=(500, 500)):
    return Sketch()


def morph(size=(500, 500)):
    sketch = Sketch()

    circle = Circle()
    circle.radius.lock(200)
    circle.center.lock()

    box = CenteredBox()
    box.position.bind("mouse_x", "mouse_y")
    box.rotation.lock()
    box.width.lock(300)
    box.height.lock(200)

    morph = Morph(circle, box)
    morph.t.lock(0.5)

    sketch.add(morph)

    return sketch


def iceberg(size=(500, 500)):
    sketch = Sketch()

    # Let's generate a random polygon
    radius = 200
    n = 5
    angles = random.sample(range(0, 360), n)
    radii = [random.uniform(0.5, 1.0) * radius for _ in range(n)]
    points = [
        (
            r * math.cos(math.radians(a)),
            r * math.sin(math.radians(a)),
        )
        for r, a in zip(radii, sorted(angles))
    ]

    # It might be fun to use a var here
    density = 0.5

    # This is for debugging, if you want to show the start shape
    """
    p0 = Polygon()
    for x, y in points:
        p = p0.add_point()
        p.lock(x, y)
    p0.rotation.lock()
    p0.position.lock()
    sketch.add(p0)
    """

    polygon = Polygon()
    for x, y in points:
        p = polygon.add_point()
        p.lock(x, y)
    sketch.add(polygon)

    center_of_gravity = Centroid(polygon, size)
    area = Area(polygon, size)

    # This is definitely a big hack to be able to have some small math with values
    mass = MathValue(area, lambda a: a * density)

    water = BottomHalfPlane()
    water.position.lock()
    water.rotation.lock()

    under_water_shape = Intersection(
        polygon,
        water,
    )
    buoyancy_center = Centroid(under_water_shape, size)
    under_water_area = Area(under_water_shape, size)

    buoyancy_circle = Circle()
    buoyancy_circle.radius.lock(2)
    buoyancy_circle.center = buoyancy_center

    sketch.add(buoyancy_circle)

    # I wrote this constraint for the specific case - to see how we could generalize this (or not)
    rotation_stable = VerticallyAligned(center_of_gravity, buoyancy_center)
    sketch.add_constraint(rotation_stable)

    # Same here
    archimedes_stable = SameValueConstraint(mass, under_water_area)
    sketch.add_constraint(archimedes_stable)

    return sketch


scene_dict = {
    "Empty": empty,
    "Chain": load_chain,
    "Morph": morph,
    "Iceberg": iceberg,
}
