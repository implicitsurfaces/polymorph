from polymorph_app.sketch import (
    CenteredBox,
    Circle,
    DistanceConstraint,
    LengthValue,
    Morph,
    Sketch,
)


def load_chain():
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


def empty():
    return Sketch()


def morph():
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


scene_dict = {
    "Empty": empty,
    "Chain": load_chain,
    "Morph": morph,
}
