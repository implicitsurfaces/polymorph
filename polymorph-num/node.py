from dataclasses import dataclass
from enum import Enum

def as_node(x):
  if(isinstance(x, Node)):
    return x
  elif(isinstance(x, float)):
    return Scalar(x)
  elif(isinstance(x, int)):
    return Scalar(float(x))
  
  raise ValueError()

class BinOp(Enum):
  Mul = "mul"
  Add = "add"
  Div = "div"
  Sub = "sub"

class UnOp(Enum):
  Sqrt = "sqrt"
  Sigmoid = "sigmoid"

class Node:
  dim: int
  
  def __init__(self, dim):
    object.__setattr__(self, "dim", dim)

  def __mul__(self, other):
    return self.__binary(other, BinOp.Mul)

  def __add__(self, other):
    return self.__binary(other, BinOp.Add)

  def __sub__(self, other):
    return self.__binary(other, BinOp.Sub)
  
  def __truediv__(self, other):
    return self.__binary(other, BinOp.Div)

  def __binary(self, other, op):
    o = as_node(other)
    if(self.dim == o.dim):
      return Binary(self, o, op)
    elif(self.dim == 1):
        return Binary(Broadcast(self, o.dim), o, op)
    elif(o.dim == 1):
        return Binary(self, Broadcast(o, self.dim), op)
    
    raise ValueError()

@dataclass(frozen=True)
class Param(Node):
  id: int

  def __post_init__(self):
    super().__init__(1)

@dataclass(frozen=True)
class Observation(Node):
  name: str

  def __post_init__(self):
    super().__init__(1)

@dataclass(frozen=True)
class Scalar(Node):
  value: float

  def __post_init__(self):
    super().__init__(1)

@dataclass(frozen=True)
class Vector(Node):
  value: list[float]

  def __post_init__(self):
    super().__init__(len(self.value))

@dataclass(frozen=True)
class Binary(Node):
  left: Node
  right: Node
  op: BinOp

  def __post_init__(self):
    if(self.left.dim == self.right.dim):
      super().__init__(self.left.dim)
    else:
      raise ValueError()

@dataclass(frozen=True)
class Unary(Node):
  orig: Node
  op: UnOp

  def __post_init__(self):
    super().__init__(self.orig.dim)

@dataclass(frozen=True)
class Broadcast(Node):
  orig: Node
  dim: int

@dataclass(frozen=True)
class Sum(Node):
  orig: Node

  def __post__init(self):
    super().__init__(1)
