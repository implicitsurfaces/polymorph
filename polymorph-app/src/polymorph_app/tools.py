import logging
import time

import glfw
import imgui
from polymorph_app.types import WorldPos
from polymorph_num.unit import CompiledUnit
from polymorph_num.util import log_perf

from .sketch import (
    Box,
    Circle,
    Constraint,
    OnBoundaryConstraint,
    PointValue,
    Polygon,
    Sketch,
)

ui_event_log = logging.getLogger("ui_event")
gesture_start_time = time.monotonic()


class Tool:
    def __init__(self, view_model):
        self.view_model = view_model

    def handle_frame(self):
        global gesture_start_time

        delta_ms = imgui.get_io().delta_time * 1000
        if delta_ms > 500:
            ui_event_log.debug(f"Frame delta: {delta_ms:.3f}ms")

        mouse_pos = self.view_model.cursor_world
        time_ms = (time.monotonic() - gesture_start_time) * 1000

        # These are defined as functions just to allow for easy logging.
        @log_perf(ui_event_log)
        def dispatch_mousedown():
            self.mousedown(mouse_pos)

        @log_perf(ui_event_log)
        def dispatch_mouseup():
            self.mouseup(mouse_pos)

        @log_perf(ui_event_log)
        def dispatch_escape():
            self.escape()

        # Handle mouse
        if imgui.is_mouse_clicked():
            gesture_start_time = time.monotonic()
            ui_event_log.debug("Dispatching mousedown t=0ms")
            dispatch_mousedown()
            self.view_model.sketch.changed()
        elif imgui.is_mouse_released():
            ui_event_log.debug(f"Dispatching mouseup at t={time_ms:.3f}ms")
            dispatch_mouseup()
            self.view_model.sketch.changed()

        # Handle keyboard events
        if imgui.is_key_pressed(glfw.KEY_ESCAPE):
            ui_event_log.debug(f"Dispatching escape at t={time_ms:.3f}ms")
            dispatch_escape()

    # Methods that can be implemented by subclasses:

    def mousedown(self, pos):
        pass

    def mouseup(self, pos):
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

    def mouseup(self, pos: WorldPos, current_unit: CompiledUnit):
        # Lock in the radius at its current value.
        self.sketch.remove_constraint(self.constraint)
        new_radius = current_unit.run(self.circle.radius.as_expr())
        self.circle.radius.lock(new_radius)


class CircleTool(Tool):
    gesture: CircleGesture | None

    def mousedown(self, pos: WorldPos):
        self.gesture = CircleGesture(self.view_model.sketch, pos)

    def mouseup(self, pos):
        if self.gesture is not None:
            self.gesture.mouseup(pos, self.view_model.current_unit)
        self.gesture = None


class BoxTool(Tool):
    box: Box | None = None

    def mousedown(self, pos):
        box = Box()
        box.p1.lock(pos.x, pos.y)
        box.p2.bind("mouse_x", "mouse_y")
        box.position.lock()
        box.rotation.lock()

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
        poly.position.lock()
        poly.rotation.lock()
        poly.temp_point = PointValue().bind("mouse_x", "mouse_y")

        sketch.add(poly)

        self.poly = poly
        self.sketch = sketch

    def mousedown(self, pos: WorldPos):
        assert self.poly is not None
        self.poly.add_point().lock(pos.x, pos.y)

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
