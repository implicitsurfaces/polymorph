from . import node


class Point:
    x: node.Node
    y: node.Node

    def __init__(self, x, y):
        self.x = node.as_node(x)
        self.y = node.as_node(y)
