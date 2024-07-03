import glfw
import imgui
import jax.numpy as jnp
from polymorph_num.expr import Observation
from polymorph_num.ops import param
from polymorph_num.vec import Vec2

from .graph import Box, Circle, Polygon


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
            self.view_model.graph.changed()  # TODO: Find a cleaner way to do this.
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
        r = param()
        self.shape = self.view_model.graph.add(Circle(Vec2(pos.x, pos.y), r))

    def mouseup(self, pos):
        # Lock in the current radius
        self.shape.radius = (Vec2(pos.x, pos.y) - self.shape.center).norm()
        self.view_model.graph.changed()  # Ugh
        self.shape = None


class BoxTool(Tool):
    def mousedown(self, pos):
        p1 = Vec2(pos.x, pos.y)
        p2 = Vec2(Observation("mouse_x"), Observation("mouse_y"))
        self.shape = self.view_model.graph.add(Box(p1, p2))

    def mouseup(self, pos):
        self.shape.p2 = Vec2(pos.x, pos.y)
        self.view_model.graph.changed()  # Ugh
        self.shape = None


def unique_point(p, other):
    """Add jitter to `p` if it's equal to `other`."""
    return p + 0.1 if jnp.array_equal(p, other) else p


class PolygonTool(Tool):
    def _add_point(self, pos):
        last_p = self.shape.points[-1] if self.shape.points else []
        self.shape.points.append(unique_point(pos, last_p))

    def mousedown(self, pos):
        if self.shape is None:
            self.shape = self.view_model.graph.add(Polygon())
        else:
            self.shape.points.pop()  # Remove the preview point.

        # Push two points: one is permanent, the other is a "preview" point.
        self._add_point(pos)
        self._add_point(pos)

    def mousemove(self, pos):
        if self.shape:
            # Update the preview point.
            self.shape.points[-1] = unique_point(pos, self.shape.points[-2])

    def escape(self):
        self.shape.points.pop()  # Remove the preview point.
        self.shape = None

    def render_feedback(self, view_model, draw_list):
        if self.shape and len(self.shape.points) < 3:
            color = imgui.get_color_u32_rgba(1, 1, 1, 1)
            draw_list.add_polyline(
                [view_model.world_to_screen(p) for p in self.shape.points],
                color,
                flags=imgui.DRAW_NONE,
                thickness=1,
            )
