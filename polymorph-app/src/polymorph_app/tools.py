import glfw

from .graph import Circle, Polygon, Rect


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

    def handle_frame(self, mouse_pos):
        if self._mousedown_pos is not None:
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


class CircleTool(Tool):
    def mousedown(self, pos):
        self.shape = self.view_model.graph.add(Circle)

    def mousedrag(self, pos, start_pos):
        self.shape.adjust(start_pos, pos)

    def mouseup(self, pos):
        self.shape = None


class RectTool(Tool):
    def mousedown(self, pos):
        self.shape = self.view_model.graph.add(Rect)

    def mousedrag(self, pos, start_pos):
        self.shape.adjust(start_pos, pos)

    def mouseup(self, pos):
        self.shape = None


class PolygonTool(Tool):
    _num_points = 0

    def mousedown(self, pos):
        if self.shape is None:
            self.shape = self.view_model.graph.add(Polygon)
        self._num_points += 1
        self.shape.points[self._num_points :] = [pos]

    def mousemove(self, pos):
        if self.shape:
            self.shape.points[self._num_points :] = [pos]

    def escape(self):
        self.shape = None
