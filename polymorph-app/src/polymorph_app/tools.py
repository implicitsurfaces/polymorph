import math

import glfw
import imgui
from polymorph_app.types import WorldPos

from .sketch import (
    Box,
    Circle,
    Constraint,
    LengthValue,
    LockedAtom,
    OnBoundaryConstraint,
    PointValue,
    Polygon,
    Sketch,
)


class Tool:
    _mousedown_pos = None

    def __init__(self, view_model):
        self.view_model = view_model

    def handle_mouse_button(self, pos, action):
        if action == glfw.PRESS:
            self._mousedown_pos = pos
            self.mousedown(pos)
        else:
            self._mousedown_pos = None
            self.mouseup(pos)

    def handle_key(self, key, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            self.escape()

    def handle_frame(self):
        mouse_pos = self.view_model.cursor_world
        if self._mousedown_pos is not None:
            self.view_model.sketch.changed()  # TODO: Find a cleaner way to do this.
            self.mousedrag(mouse_pos, self._mousedown_pos)
        else:
            self.mousemove(mouse_pos)

    # Methods that can be implemented by subclasses:

    def mousedown(self, pos):
        pass

    def mousedrag(self, pos, start_pos):
        pass

    def mouseup(self, pos):
        pass

    def mousemove(self, pos):
        pass

    def escape(self):
        pass

    def render_feedback(self, view_model, draw_list):
        pass


class CircleGesture:
    sketch: Sketch
    circle: Circle
    constraint: Constraint

    def __init__(self, sketch: Sketch, pos: WorldPos):
        circle = Circle()
        circle.center.lock(pos.x, pos.y)
        bp = circle.boundary_point().bind("mouse_x", "mouse_y")
        self.constraint = OnBoundaryConstraint(circle, bp)

        sketch.add(circle)
        sketch.add_constraint(self.constraint)

        self.circle = circle
        self.sketch = sketch

    def _distance(self, center: PointValue, pos: WorldPos):
        match center:
            case PointValue(LockedAtom(float(x)), LockedAtom(float(y))):
                # Lock in the current radius.
                return math.sqrt((x - pos.x) ** 2 + (y - pos.y) ** 2)
        raise ValueError(f"Not a PointValue: {center}")

    def mouseup(self, pos: WorldPos):
        # Lock in the radius at its current value.
        self.sketch.remove_constraint(self.constraint)
        r = self._distance(self.circle.center, pos)
        self.circle.radius = LengthValue().lock(r)


class CircleTool(Tool):
    gesture: CircleGesture | None

    def mousedown(self, pos: WorldPos):
        self.gesture = CircleGesture(self.view_model.sketch, pos)

    def mouseup(self, pos):
        if self.gesture is not None:
            self.gesture.mouseup(pos)
        self.gesture = None


class BoxTool(Tool):
    box: Box | None = None

    def mousedown(self, pos):
        p1 = PointValue().lock(pos.x, pos.y)
        p2 = PointValue().bind("mouse_x", "mouse_y")
        box = Box(p1, p2)
        self.view_model.sketch.add(box)
        self.box = box

    def mouseup(self, pos):
        assert self.box is not None
        p2 = PointValue().lock(pos.x, pos.y)
        self.box.p2 = p2
        self.view_model.sketch.changed()  # Ugh
        self.box = None


class PolygonGesture:
    sketch: Sketch
    poly: Polygon

    def __init__(self, sketch: Sketch, pos: WorldPos):
        poly = Polygon()
        sketch.add(poly)

        self.poly = poly
        self.sketch = sketch

    def mousedown(self, pos: WorldPos):
        assert self.poly is not None
        self.poly.points.append(PointValue().lock(pos.x, pos.y))

    def mousemove(self, pos: WorldPos):
        self.poly.temp_point = PointValue().bind("mouse_x", "mouse_y")

    def mouseup(self, pos: WorldPos):
        self.poly.temp_point = None

    def end(self):
        self.poly.temp_point = None
        self.sketch.changed()  # Ugh


class PolygonTool(Tool):
    gesture: PolygonGesture | None
    raw_points: list[WorldPos]

    def __init__(self, view_model):
        super().__init__(view_model)
        self.gesture = None
        self.raw_points = []

    def mousedown(self, pos):
        if self.gesture is None:
            self.gesture = PolygonGesture(self.view_model.sketch, pos)
        self.gesture.mousedown(pos)
        self.raw_points.append(pos)

    def mousemove(self, pos):
        if self.gesture:
            self.gesture.mousemove(pos)

    def escape(self):
        if self.gesture:
            self.gesture.end()
            self.gesture = None

    def render_feedback(self, view_model, draw_list):
        if self.gesture:
            color = imgui.get_color_u32_rgba(1, 1, 1, 1)
            points = self.raw_points + [view_model.cursor_world]
            draw_list.add_polyline(
                [view_model.world_to_screen(p) for p in points],
                color,
                flags=imgui.DRAW_NONE,
                thickness=1,
            )
