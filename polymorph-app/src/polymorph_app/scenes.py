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
    LockedAtom,
    MathValue,
    Morph,
    PointValue,
    Polygon,
    SameValueConstraint,
    Sketch,
    VerticallyAligned,
)


class Scene:
    name: str
    vars: dict[str, float]
    actions: list[str]

    def __init__(
        self,
        name: str,
        vars: dict[str, float] | None = None,
        actions: list[str] | None = None,
    ):
        """
        name -- Name of the scene (e.g. to show in the UI)
        vars -- user-editable values for the scene (will be passed to the
                the minimizer as observations)
        actions -- method names that should be exposed as buttons in the UI
        """
        self.name = name
        self.vars = vars if vars is not None else {}
        self.actions = actions if actions is not None else []


class ChainScene(Scene):
    def __init__(self):
        super().__init__("Chain", {})

    def load(self):
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


class EmptyScene(Scene):
    def __init__(self):
        super().__init__("Empty")

    def load(self):
        return Sketch()


class MorphScene(Scene):
    def __init__(self):
        super().__init__("Morph", {"box share": 0.5}, ["toggle_lock_position"])
        self.box_locked = False
        self.box: None | CenteredBox = None

    def load(self):
        sketch = Sketch()

        circle = Circle()
        circle.radius.lock(200)
        circle.center.lock()

        box = CenteredBox()
        box.position.bind("mouse_x", "mouse_y")
        box.rotation.lock()
        box.width.lock(300)
        box.height.lock(200)

        self.box = box
        self.box_locked = False

        morph = Morph(circle, box)
        morph.t.bind("box share")

        sketch.add(morph)

        return sketch

    def toggle_lock_position(self, sketch: Sketch):
        if self.box is None:
            return sketch

        if not self.box_locked:
            self.box.position.lock(0, 0)
            self.box_locked = True
        else:
            self.box.position.bind("mouse_x", "mouse_y")
            self.box_locked = False

        sketch.changed()

        return sketch


class IcebergScene(Scene):
    def __init__(self):
        super().__init__("Iceberg", {"density": 0.5}, ["run", "reset"])

    def load(self):
        return Sketch()

    def reset(self, sketch: Sketch):
        sketch.reset()

    def _add_random_polygon(self, sketch: Sketch):
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

        polygon = Polygon()
        for x, y in points:
            p = polygon.add_point()
            p.lock(x, y)
        sketch.add(polygon)
        return polygon

    def run(self, sketch: Sketch):
        # Find a user-created polygon, or generate a random one.
        polygons = [s for s in sketch.shapes if isinstance(s, Polygon)]
        assert len(polygons) <= 1
        polygon = polygons[0] if polygons else self._add_random_polygon(sketch)
        polygon.position.free()
        polygon.rotation.free()

        # Show the start shape for debugging
        p0 = Polygon()
        for point_val in polygon.points:
            match point_val:
                case PointValue(LockedAtom(x), LockedAtom(y)):
                    p0.add_point().lock(x, y)
                case _:
                    raise Exception(f"Not a locked PointValue: {point_val}")
        p0.rotation.lock()
        p0.position.lock()
        sketch.add(p0)

        center_of_gravity = Centroid(polygon)
        area = Area(polygon)

        # This is definitely a big hack to be able to have some small math with values
        mass = MathValue(area, lambda a: a * 0.5)

        water = BottomHalfPlane()
        water.position.lock()
        water.rotation.lock()

        under_water_shape = Intersection(
            polygon,
            water,
        )
        buoyancy_center = Centroid(under_water_shape)
        under_water_area = Area(under_water_shape)

        sketch.add(under_water_shape)

        # I wrote this constraint for the specific case - to see how we could generalize this (or not)
        rotation_stable = VerticallyAligned(center_of_gravity, buoyancy_center)
        sketch.add_constraint(rotation_stable)

        # Same here
        archimedes_stable = SameValueConstraint(mass, under_water_area)
        sketch.add_constraint(archimedes_stable)

        return sketch


scene_dict = {
    "Empty": EmptyScene(),
    "Chain": ChainScene(),
    "Morph": MorphScene(),
    "Iceberg": IcebergScene(),
}
