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

class Node:
  def __init__(self, dim):
    self.dim = dim

  def __mul__(self, other):
    return self.__binary(other, BinOp.Mul)

  def __add__(self, other):
    return self.__binary(other, BinOp.Add)

  def __sub__(self, other):
    return self.__binary(other, BinOp.Sub)
  
  def __div__(self, other):
    return self.__binary(other, BinOp.Div)

  def __binary(self, other, op):
    if(self.dim == other.dim):
      return Binary(self, as_node(other), op)
    elif(self.dim == 1):
        return Binary(Broadcast(self, other.dim), other, op)
    elif(other.dim == 1):
        return Binary(self, Broadcast(other, self.dim), op)
    
    raise ValueError()

@dataclass
class Var(Node):
  position: int

  def __post_init__(self):
    super().__init__(1)

@dataclass
class Scalar(Node):
  value: float

  def __post_init__(self):
    super().__init__(1)

@dataclass
class Vector(Node):
  value: list[float]

  def __post_init__(self):
    super().__init__(len(self.value))

@dataclass
class Binary(Node):
  left: Node
  right: Node
  op: BinOp

  def __post_init__(self):
    if(self.left.dim == self.right.dim):
      super().__init__(self.left.dim)
    else:
      raise ValueError()

@dataclass
class Unary(Node):
  orig: Node
  op: UnOp

  def __post_init__(self):
    super().__init__(self.orig.dim)

@dataclass
class Broadcast(Node):
  orig: Node
  dim: int

@dataclass
class Sum(Node):
  orig: Node

  def __post__init(self):
    super().__init__(1)
