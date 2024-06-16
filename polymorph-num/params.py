import node as n

class ParamMap:
    def __init__(self):
        self.count = 0
        self.dict = dict()
    
    def add(self, node):
        if(node not in self.dict):
            self.dict[node] = self.count
            self.count += 1

    def get(self, node, values):
        return values[self.dict[node]]

def find_params(node):
    map = ParamMap()
    _find_params(node, map)
    return map

def _find_params(node, map):
    match node:
        case n.Broadcast(orig, dim):
            _find_params(orig, map)
        
        case n.Unary(orig, op):
            _find_params(orig, map)
                
        case n.Binary(left, right, op):
            _find_params(left, map)
            _find_params(right, map)
                
        case n.Param(pos):
            map.add(node)
        
        case n.Sum(orig):
            _find_params(orig, map)
