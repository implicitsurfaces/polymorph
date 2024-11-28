import {
  BinaryOp,
  BinaryOperation,
  LiteralNum,
  NEG_ONE_NODE,
  NumNode,
  ONE_NODE,
  TWO_NODE,
  UnaryOp,
  Variable,
  Derivative,
  ZERO_NODE,
} from "./num-tree";

function nodeAnd(lhs: NumNode, rhs: NumNode): NumNode {
  return new BinaryOp("AND", lhs, rhs);
}

function nodeOr(lhs: NumNode, rhs: NumNode): NumNode {
  return new BinaryOp("OR", lhs, rhs);
}

function nodeNot(a: NumNode): NumNode {
  return new UnaryOp("NOT", a);
}

function nodeMax(a: NumNode, b: NumNode): NumNode {
  return new BinaryOp("MAX", a, b);
}

function nodeMin(a: NumNode, b: NumNode): NumNode {
  return new BinaryOp("MIN", a, b);
}

function nodeCompare(a: NumNode, b: NumNode): NumNode {
  return new BinaryOp("COMPARE", a, b);
}

function nodeLessThan(a: NumNode, b: NumNode): NumNode {
  return nodeMax(ZERO_NODE, nodeCompare(b, a));
}

function nodeIfTruthyElse(
  condition: NumNode,
  ifNonZero: NumNode,
  ifZero: NumNode,
): NumNode {
  const lhs = nodeAnd(condition, ifNonZero);
  const n_condition = nodeNot(condition);
  const rhs = nodeAnd(n_condition, ifZero);
  return nodeOr(lhs, rhs);
}

function unaryDerivative(operation: string, operand: NumNode): NumNode {
  if (operation === "SQRT") {
    return new BinaryOp(
      "DIV",
      ONE_NODE,
      new BinaryOp("MUL", TWO_NODE, new UnaryOp("SQRT", operand)),
    );
  }

  if (operation === "COS") {
    return new UnaryOp("NEG", new UnaryOp("SIN", operand));
  }

  if (operation === "SIN") {
    return new UnaryOp("COS", operand);
  }

  if (operation === "TAN") {
    return new BinaryOp(
      "DIV",
      ONE_NODE,
      new BinaryOp(
        "MUL",
        new UnaryOp("COS", operand),
        new UnaryOp("COS", operand),
      ),
    );
  }

  if (operation === "ACOS") {
    return new UnaryOp(
      "NEG",
      new BinaryOp(
        "DIV",
        ONE_NODE,
        new UnaryOp(
          "SQRT",
          new BinaryOp("SUB", ONE_NODE, new BinaryOp("MUL", operand, operand)),
        ),
      ),
    );
  }

  if (operation === "ASIN") {
    return new BinaryOp(
      "DIV",
      ONE_NODE,
      new UnaryOp(
        "SQRT",
        new BinaryOp("SUB", ONE_NODE, new BinaryOp("MUL", operand, operand)),
      ),
    );
  }

  if (operation === "ATAN") {
    return new BinaryOp(
      "DIV",
      ONE_NODE,
      new BinaryOp("ADD", ONE_NODE, new BinaryOp("MUL", operand, operand)),
    );
  }

  if (operation === "EXP") {
    return new UnaryOp("EXP", operand);
  }

  if (operation === "LOG") {
    return new BinaryOp("DIV", ONE_NODE, operand);
  }

  if (operation === "ABS") {
    return new UnaryOp("SIGN", operand);
  }

  if (operation === "NEG") {
    return NEG_ONE_NODE;
  }

  if (operation === "LOG1P") {
    return new BinaryOp(
      "DIV",
      ONE_NODE,
      new BinaryOp("ADD", ONE_NODE, operand),
    );
  }

  if (operation === "TANH") {
    const twoCosh = new BinaryOp(
      "ADD",
      new UnaryOp("EXP", operand),
      new UnaryOp("EXP", new UnaryOp("NEG", operand)),
    );

    return new BinaryOp(
      "DIV",
      new LiteralNum(4),
      new BinaryOp("MUL", twoCosh, twoCosh),
    );
  }

  if (operation === "SIGN") {
    return ZERO_NODE;
  }
  if (operation === "NOT") {
    return ZERO_NODE;
  }

  throw new Error(`Unknown unary operation for derivation: ${operation}`);
}

function binaryDerivative(
  operation: BinaryOperation,
  left: NumNode,
  right: NumNode,
): NumNode {
  const leftDerivative = fullDerivative(left);
  const rightDerivative = fullDerivative(right);

  if (operation === "ADD" || operation === "SUB") {
    return new BinaryOp(operation, leftDerivative, rightDerivative);
  }

  if (operation === "MUL") {
    return new BinaryOp(
      "ADD",
      new BinaryOp("MUL", leftDerivative, right),
      new BinaryOp("MUL", left, rightDerivative),
    );
  }

  if (operation === "DIV") {
    return new BinaryOp(
      "DIV",
      new BinaryOp(
        "SUB",
        new BinaryOp("MUL", leftDerivative, right),
        new BinaryOp("MUL", left, rightDerivative),
      ),
      new BinaryOp("MUL", right, right),
    );
  }

  if (operation === "ATAN2") {
    const leftSquared = new BinaryOp("MUL", left, left);
    const rightSquared = new BinaryOp("MUL", right, right);
    const sumSquared = new BinaryOp("ADD", leftSquared, rightSquared);
    return new BinaryOp(
      "DIV",
      new BinaryOp(
        "SUB",
        new BinaryOp("MUL", right, leftDerivative),
        new BinaryOp("MUL", left, rightDerivative),
      ),
      sumSquared,
    );
  }

  if (operation === "MOD") {
    throw new Error("Derivative of MOD is not implemented");
  }

  if (operation === "MAX") {
    const isLeftGreater = nodeLessThan(right, left);
    return nodeIfTruthyElse(isLeftGreater, leftDerivative, rightDerivative);
  }

  if (operation === "MIN") {
    const isLeftLess = nodeLessThan(left, right);
    return nodeIfTruthyElse(isLeftLess, leftDerivative, rightDerivative);
  }

  if (operation === "AND") {
    // if left is 0 then return left branch else return right branch
    return nodeIfTruthyElse(left, rightDerivative, leftDerivative);
  }

  if (operation === "OR") {
    // if left is 0 then return right branch else return left branch
    return nodeIfTruthyElse(left, leftDerivative, rightDerivative);
  }

  throw new Error(`Unknown binary operation for derivation: ${operation}`);
}

export function fullDerivative(node: NumNode): NumNode {
  if (node instanceof Variable) {
    return new Derivative(node);
  }

  if (node instanceof LiteralNum) {
    return ZERO_NODE;
  }

  if (node instanceof UnaryOp) {
    if (node.operation === "NOT" || node.operation === "SIGN") {
      return ZERO_NODE;
    }

    const innerDerivative = fullDerivative(node.original);
    return new BinaryOp(
      "MUL",
      unaryDerivative(node.operation, node.original),
      innerDerivative,
    );
  }

  if (node instanceof BinaryOp) {
    if (node.operation === "COMPARE") {
      return ZERO_NODE;
    }
    return binaryDerivative(node.operation, node.left, node.right);
  }

  throw new Error(`Unknown node type for derivation: ${node}`);
}
