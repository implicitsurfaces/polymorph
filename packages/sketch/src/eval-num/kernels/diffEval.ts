import { BinaryOperation, UnaryOperation } from "../../types";
import { NumEvalKernel } from "../../types";

import {
  BinaryOp,
  LiteralNum,
  NEG_ONE_NODE,
  ONE_NODE,
  TWO_NODE,
  UnaryOp,
  Variable,
  Derivative,
  ZERO_NODE,
  NumNode,
} from "../../num-tree";

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

export class DiffEvalKernel implements NumEvalKernel<NumNode> {
  variable(_: string, node: NumNode) {
    return new Derivative(node as Variable);
  }
  literal() {
    return ZERO_NODE;
  }
  value(): number {
    throw new Error("Method not implemented.");
  }

  unaryOp(
    operation: UnaryOperation,
    innerDerivative: NumNode,
    node: NumNode,
  ): NumNode {
    if (operation === "NOT" || operation === "SIGN") {
      return ZERO_NODE;
    }

    const dx = (derivative: NumNode) => {
      return new BinaryOp("MUL", derivative, innerDerivative);
    };

    const operand = (node as UnaryOp).original;

    if (operation === "SQRT") {
      return dx(
        new BinaryOp(
          "DIV",
          ONE_NODE,
          new BinaryOp("MUL", TWO_NODE, new UnaryOp("SQRT", operand)),
        ),
      );
    }

    if (operation === "COS") {
      return dx(new UnaryOp("NEG", new UnaryOp("SIN", operand)));
    }

    if (operation === "SIN") {
      return dx(new UnaryOp("COS", operand));
    }

    if (operation === "TAN") {
      return dx(
        new BinaryOp(
          "DIV",
          ONE_NODE,
          new BinaryOp(
            "MUL",
            new UnaryOp("COS", operand),
            new UnaryOp("COS", operand),
          ),
        ),
      );
    }

    if (operation === "ACOS") {
      return dx(
        new UnaryOp(
          "NEG",
          new BinaryOp(
            "DIV",
            ONE_NODE,
            new UnaryOp(
              "SQRT",
              new BinaryOp(
                "SUB",
                ONE_NODE,
                new BinaryOp("MUL", operand, operand),
              ),
            ),
          ),
        ),
      );
    }

    if (operation === "ASIN") {
      return dx(
        new BinaryOp(
          "DIV",
          ONE_NODE,
          new UnaryOp(
            "SQRT",
            new BinaryOp(
              "SUB",
              ONE_NODE,
              new BinaryOp("MUL", operand, operand),
            ),
          ),
        ),
      );
    }

    if (operation === "ATAN") {
      return dx(
        new BinaryOp(
          "DIV",
          ONE_NODE,
          new BinaryOp("ADD", ONE_NODE, new BinaryOp("MUL", operand, operand)),
        ),
      );
    }

    if (operation === "EXP") {
      return dx(new UnaryOp("EXP", operand));
    }

    if (operation === "LOG") {
      return dx(new BinaryOp("DIV", ONE_NODE, operand));
    }

    if (operation === "ABS") {
      return dx(new UnaryOp("SIGN", operand));
    }

    if (operation === "NEG") {
      return dx(NEG_ONE_NODE);
    }

    if (operation === "LOG1P") {
      return dx(
        new BinaryOp("DIV", ONE_NODE, new BinaryOp("ADD", ONE_NODE, operand)),
      );
    }

    if (operation === "TANH") {
      const twoCosh = new BinaryOp(
        "ADD",
        new UnaryOp("EXP", operand),
        new UnaryOp("EXP", new UnaryOp("NEG", operand)),
      );

      return dx(
        new BinaryOp(
          "DIV",
          new LiteralNum(4),
          new BinaryOp("MUL", twoCosh, twoCosh),
        ),
      );
    }

    if (operation === "DEBUG") {
      return innerDerivative;
    }

    if (operation === "CBRT") {
      return dx(
        new BinaryOp(
          "DIV",
          ONE_NODE,
          new BinaryOp(
            "MUL",
            new LiteralNum(3),
            new UnaryOp("CBRT", new BinaryOp("MUL", operand, operand)),
          ),
        ),
      );
    }

    throw new Error(`Unknown unary operation for derivation: ${operation}`);
  }

  binaryOp(
    operation: BinaryOperation,
    lhs: NumNode,
    rhs: NumNode,
    node: NumNode,
  ) {
    const leftDerivative = lhs;
    const rightDerivative = rhs;

    const left = (node as BinaryOp).left;
    const right = (node as BinaryOp).right;

    if (operation === "COMPARE") {
      return ZERO_NODE;
    } else if (operation === "ADD" || operation === "SUB") {
      return new BinaryOp(operation, leftDerivative, rightDerivative);
    } else if (operation === "MUL") {
      return new BinaryOp(
        "ADD",
        new BinaryOp("MUL", leftDerivative, right),
        new BinaryOp("MUL", left, rightDerivative),
      );
    } else if (operation === "DIV") {
      return new BinaryOp(
        "DIV",
        new BinaryOp(
          "SUB",
          new BinaryOp("MUL", leftDerivative, right),
          new BinaryOp("MUL", left, rightDerivative),
        ),
        new BinaryOp("MUL", right, right),
      );
    } else if (operation === "ATAN2") {
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
    } else if (operation === "MOD") {
      return ONE_NODE;
    } else if (operation === "MAX") {
      const isLeftGreater = nodeLessThan(right, left);
      return nodeIfTruthyElse(isLeftGreater, leftDerivative, rightDerivative);
    } else if (operation === "MIN") {
      const isLeftLess = nodeLessThan(left, right);
      return nodeIfTruthyElse(isLeftLess, leftDerivative, rightDerivative);
    } else if (operation === "AND") {
      // if left is 0 then return left branch else return right branch
      return nodeIfTruthyElse(left, rightDerivative, leftDerivative);
    } else if (operation === "OR") {
      // if left is 0 then return right branch else return left branch
      return nodeIfTruthyElse(left, leftDerivative, rightDerivative);
    } else {
      throw new Error(`Unknown binary operation for derivation: ${operation}`);
    }
  }
}
