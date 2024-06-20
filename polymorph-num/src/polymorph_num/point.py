from . import node, ops


class Point:
    x: node.Node
    y: node.Node

    __match_args__ = ("x", "y")

    @classmethod
    def origin(cls):
        return cls(0.0, 0.0)

    def __init__(self, x, y):
        self.x = node.as_node(x)
        self.y = node.as_node(y)

    def __sub__(self, other):
        return Vec2(self.x - other.x, self.y - other.y)

    def __add__(self, other):
        match other:
            case Vec2(x, y):
                return Point(self.x + x, self.y + y)
            case Point(x, y):
                return Vec2(self.x + other.x, self.y + other.y)


class Vec2:
    x: node.Node
    y: node.Node

    __match_args__ = ("x", "y")

    def __init__(self, x, y):
        self.x = node.as_node(x)
        self.y = node.as_node(y)

    def __truediv__(self, other):
        return Vec2(self.x / node.as_node(other), self.y / node.as_node(other))

    def length(self):
        return ops.sqrt(self.x * self.x + self.y * self.y)
