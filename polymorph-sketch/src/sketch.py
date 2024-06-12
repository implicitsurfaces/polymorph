import types as t


class Sketch:
    def __init__(self):
        self.points = []
        self.dimensions = []
        self.constraints = []

    def point(self):
        p = t.Point()
        self.points.append(p)
        return p

    def length(self):
        l = t.Dimension("mm")
        self.dimensions.append(l)
        return l

    def angle(self):
        a = t.Dimension("rad")
        self.dimensions.append(a)
        return a

    def distance(self, a, b, l):
        c = t.Distance(a, b, l)
        self.constraints.append(c)
        return c

    def direction(self, a, b, theta):
        c = t.Direction(a, b, theta)
        self.constraints.append(c)
        return c

    def sum(self, a, b, res):
        c = t.Sum(a, b, res)
        self.constraints.append(c)
        return c

    def difference(self, a, b, res):
        c = t.Difference(a, b, res)
        self.constraints.append(c)
        return c
