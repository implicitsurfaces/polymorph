from polymorph_app.sketch import Circle, DistanceConstraint, LengthValue, Sketch


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


scene_dict = {
    "Empty": empty,
    "Chain": load_chain,
}
