import glfw
import imgui
from polymorph_app.types import WorldPos
from polymorph_num.expr import Observation
from polymorph_num.vec import Vec2

from .sketch import Box, Circle, OnBoundaryConstraint, Polygon


class Tool:
    _mousedown_pos = None
    shape = None

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


class CircleTool(Tool):
    def mousedown(self, pos):
        self.shape = self.view_model.sketch.add(Circle())
        self.shape.center.lock(pos.x, pos.y)
        self.bp = self.shape.boundary_point()
        self.view_model.sketch.constraints.append(
            OnBoundaryConstraint(self.shape, self.bp)
        )
        self.bp.bind("mouse_x", "mouse_y")

    def mouseup(self, pos):
        # Lock in the current radius
        assert self.shape is not None
        center = self.shape.center.as_vec2()
        self.shape.radius.lock((Vec2(pos.x, pos.y) - center).norm())
        self.view_model.sketch.changed()  # Ugh
        self.shape = None


class BoxTool(Tool):
    def mousedown(self, pos):
        p1 = Vec2(pos.x, pos.y)
        p2 = Vec2(Observation("mouse_x"), Observation("mouse_y"))
        self.shape = self.view_model.sketch.add(Box(p1, p2))

    def mouseup(self, pos):
        self.shape.p2 = Vec2(pos.x, pos.y)
        self.view_model.sketch.changed()  # Ugh
        self.shape = None


class PolygonTool(Tool):
    points: list[WorldPos]

    def __init__(self, view_model):
        super().__init__(view_model)
        self.points = []

    def mousedown(self, pos):
        if self.shape is None:
            self.shape = self.view_model.sketch.add(Polygon())
            self.shape.temp_point = Vec2(Observation("mouse_x"), Observation("mouse_y"))
        self._add_point(pos)

    def _add_point(self, pos):
        assert self.shape is not None
        self.points.append(pos)
        self.shape.points = [Vec2(pos.x, pos.y) for pos in self.points]
        self.view_model.sketch.changed()  # Ugh

    def escape(self):
        assert self.shape is not None
        self.shape.temp_point = None
        self.shape = None
        self.view_model.sketch.changed()  # Ugh

    def render_feedback(self, view_model, draw_list):
        if self.shape:
            color = imgui.get_color_u32_rgba(1, 1, 1, 1)
            points = self.points + [view_model.cursor_world]
            draw_list.add_polyline(
                [view_model.world_to_screen(p) for p in points],
                color,
                flags=imgui.DRAW_NONE,
                thickness=1,
            )
