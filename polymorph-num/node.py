from dataclasses import dataclass
from enum import Enum

def as_node(x):
  if(isinstance(x, Node)):
    return x
  else:
    return Constant(float(x))

class BinOp(Enum):
  Mul = "mul"
  Add = "add"
  Div = "div"
  Sub = "sub"

class UnOp(Enum):
  Sqrt = "sqrt"
  
class Node:
  def __mul__(self, other):
    return Binary(self, as_node(other), BinOp.Mul)

  def __add__(self, other):
    return Binary(self, as_node(other), BinOp.Add)

  def __sub__(self, other):
    return Binary(self, as_node(other), BinOp.Sub)
  
  def __div__(self, other):
    return Binary(self, as_node(other), BinOp.Div)

@dataclass
class Var(Node):
  position: int

@dataclass
class Constant(Node):
  value: float

@dataclass
class Binary(Node):
  left: Node
  right: Node
  op: BinOp

@dataclass
class Unary(Node):
  orig: Node
  op: UnOp
