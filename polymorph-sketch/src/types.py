class Point:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.locked = False


class Dimension:
    def __init__(self, unit):
        self.value = 0.0
        self.unit = unit
        self.locked = False


class Constraint:
    def __init__(self):
        pass


class Distance(Constraint):
    def __init__(self, a: Point, b: Point, length: Dimension):
        self.a = a
        self.b = b
        self.length = length


class Direction(Constraint):
    def __init__(self, a: Point, b: Point, angle: Dimension):
        self.a = a
        self.b = b
        self.angle = angle


class Sum(Constraint):
    def __init__(self, a: Dimension, b: Dimension, result: Dimension):
        self.a = a
        self.b = b
        self.result = result


class Difference(Constraint):
    def __init__(self, a: Dimension, b: Dimension, result: Dimension):
        self.a = a
        self.b = b
        self.result = result
