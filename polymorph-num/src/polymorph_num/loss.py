from . import node as n


class Loss:
    def __init__(self, loss):
        self.loss = loss
        self.params = ParamMap()
        self.observations = {}
        _find_params(loss, self.params, self.observations)
        self.nodes = []

    def register_output(self, node):
        node_params = ParamMap()
        _find_params(node, node_params, {})
        extra_params = node_params.nodes() - self.params.nodes()
        if len(extra_params) > 0:
            raise ValueError()
        self.nodes.append(node)


def _find_params(node, params, observations):
    match node:
        case n.Broadcast(orig, dim):
            _find_params(orig, params, observations)

        case n.Unary(orig, op):
            _find_params(orig, params, observations)

        case n.Binary(left, right, op):
            _find_params(left, params, observations)
            _find_params(right, params, observations)

        case n.Param(id):
            params.add(node)

        case n.Observation(name):
            observations[name] = 0.0

        case n.Sum(orig):
            _find_params(orig, params, observations)


class ParamMap:
    def __init__(self):
        self.count = 0
        self.dict = dict()

    def add(self, node):
        if node not in self.dict:
            self.dict[node] = self.count
            self.count += 1

    def get(self, node, values):
        return values[self.dict[node]]

    def nodes(self):
        return set(self.dict.keys())
