import {
  NumNode,
  UnaryOp,
  BinaryOp,
  LiteralNum,
  ZERO_NODE as ZERO,
  ONE_NODE as ONE,
  DebugNode,
} from "../../num-tree"; // Assuming the types are defined in a separate file
import { UnaryOperation, BinaryOperation, NumEvalKernel } from "../../types";
import { evaluateBinaryOp, evaluateUnaryOp } from "./jsEval";

export class ConstantsFoldEval implements NumEvalKernel<NumNode> {
  variable(_: string, node: NumNode): NumNode {
    return node;
  }
  literal(_: number, node: NumNode): NumNode {
    return node;
  }
  value(): number {
    throw new Error("Method not implemented.");
  }

  unaryOp(
    operation: UnaryOperation,
    simplifiedOperand: NumNode,
    node: NumNode,
  ): NumNode {
    // If operand is a literal, we can evaluate the operation
    if (simplifiedOperand instanceof LiteralNum && node.operation !== "DEBUG") {
      return new LiteralNum(
        evaluateUnaryOp(operation, simplifiedOperand.value),
      );
    }

    // Identity and simplification rules
    switch (operation) {
      case "ABS":
        // abs(abs(x)) = abs(x)
        if (
          simplifiedOperand instanceof UnaryOp &&
          simplifiedOperand.operation === "ABS"
        ) {
          return simplifiedOperand;
        }
        break;
      case "NEG":
        // -(-x) = x
        if (
          simplifiedOperand instanceof UnaryOp &&
          simplifiedOperand.operation === "NEG"
        ) {
          return simplifiedOperand.original;
        }
        break;
      case "NOT":
        // !(!(x)) = x
        if (
          simplifiedOperand instanceof UnaryOp &&
          simplifiedOperand.operation === "NOT"
        ) {
          return simplifiedOperand.original;
        }
        break;
    }

    // If we couldn't simplify further, return a new unary op with the simplified operand
    if (simplifiedOperand !== (node as UnaryOp).original) {
      if (node instanceof DebugNode)
        return new DebugNode(simplifiedOperand, node.debug);
      return new UnaryOp(operation, simplifiedOperand);
    }

    return node;
  }

  binaryOp(
    operation: BinaryOperation,
    simplifiedLeft: NumNode,
    simplifiedRight: NumNode,
    node: NumNode,
  ): NumNode {
    // If both operands are literals, we can evaluate the operation
    if (
      simplifiedLeft instanceof LiteralNum &&
      simplifiedRight instanceof LiteralNum
    ) {
      return new LiteralNum(
        evaluateBinaryOp(
          operation,
          simplifiedLeft.value,
          simplifiedRight.value,
        ),
      );
    }

    // Handle specific cases for optimization
    switch (operation) {
      case "ADD":
        // x + 0 = x
        if (isZero(simplifiedRight)) {
          return simplifiedLeft;
        }
        // 0 + x = x
        if (isZero(simplifiedLeft)) {
          return simplifiedRight;
        }
        // Combine literal terms if one side is a literal
        if (
          simplifiedLeft instanceof LiteralNum &&
          simplifiedRight instanceof BinaryOp &&
          simplifiedRight.operation === "ADD" &&
          simplifiedRight.left instanceof LiteralNum
        ) {
          // l1 + (l2 + x) = (l1 + l2) + x
          const combinedLiteral = new LiteralNum(
            simplifiedLeft.value + simplifiedRight.left.value,
          );
          return new BinaryOp("ADD", combinedLiteral, simplifiedRight.right);
        }
        break;

      case "SUB":
        // x - 0 = x
        if (isZero(simplifiedRight)) {
          return simplifiedLeft;
        }
        // x - x = 0
        if (areNodesEqual(simplifiedLeft, simplifiedRight)) {
          return ZERO;
        }
        break;

      case "MUL":
        // x * 0 = 0
        if (isZero(simplifiedLeft) || isZero(simplifiedRight)) {
          return ZERO;
        }
        // x * 1 = x
        if (isOne(simplifiedRight)) {
          return simplifiedLeft;
        }
        // 1 * x = x
        if (isOne(simplifiedLeft)) {
          return simplifiedRight;
        }
        break;

      case "DIV":
        // x / 1 = x
        if (isOne(simplifiedRight)) {
          return simplifiedLeft;
        }
        // 0 / x = 0 (assume x != 0)
        if (isZero(simplifiedLeft)) {
          return ZERO;
        }
        // x / x = 1 (assume x != 0)
        if (areNodesEqual(simplifiedLeft, simplifiedRight)) {
          return ONE;
        }
        break;

      case "MIN":
        // min(x, x) = x
        if (areNodesEqual(simplifiedLeft, simplifiedRight)) {
          return simplifiedLeft;
        }
        // min(a, b) where a and b are comparable literals
        if (
          simplifiedLeft instanceof LiteralNum &&
          simplifiedRight instanceof LiteralNum
        ) {
          return new LiteralNum(
            Math.min(simplifiedLeft.value, simplifiedRight.value),
          );
        }
        // If we can prove one side is always smaller
        if (canProveRelation(simplifiedLeft, simplifiedRight, "LT")) {
          return simplifiedLeft;
        }
        if (canProveRelation(simplifiedRight, simplifiedLeft, "LT")) {
          return simplifiedRight;
        }
        break;

      case "MAX":
        // max(x, x) = x
        if (areNodesEqual(simplifiedLeft, simplifiedRight)) {
          return simplifiedLeft;
        }
        // max(a, b) where a and b are comparable literals
        if (
          simplifiedLeft instanceof LiteralNum &&
          simplifiedRight instanceof LiteralNum
        ) {
          return new LiteralNum(
            Math.max(simplifiedLeft.value, simplifiedRight.value),
          );
        }
        // If we can prove one side is always larger
        if (canProveRelation(simplifiedLeft, simplifiedRight, "GT")) {
          return simplifiedLeft;
        }
        if (canProveRelation(simplifiedRight, simplifiedLeft, "GT")) {
          return simplifiedRight;
        }
        break;

      case "AND":
        // false AND x = false
        if (isZero(simplifiedLeft)) {
          return ZERO;
        }
        // true AND x = x
        if (
          simplifiedLeft instanceof LiteralNum &&
          simplifiedLeft.value !== 0
        ) {
          return simplifiedRight;
        }
        break;

      case "OR":
        // true OR x = true
        if (
          simplifiedLeft instanceof LiteralNum &&
          simplifiedLeft.value !== 0
        ) {
          return simplifiedLeft;
        }
        // false OR x = x
        if (isZero(simplifiedLeft)) {
          return simplifiedRight;
        }
        break;
    }

    // If we couldn't simplify further but the operands have changed,
    // return a new binary op with the simplified operands
    if (
      simplifiedLeft !== (node as BinaryOp).left ||
      simplifiedRight !== (node as BinaryOp).right
    ) {
      return new BinaryOp(operation, simplifiedLeft, simplifiedRight);
    }

    return node;
  }
}

function isZero(node: NumNode): boolean {
  if (node === ZERO) return true;
  return node instanceof LiteralNum && node.value === 0;
}

function isOne(node: NumNode): boolean {
  if (node === ONE) return true;
  return node instanceof LiteralNum && node.value === 1;
}

/**
 * Check if two nodes are structurally equal
 */
function areNodesEqual(node1: NumNode, node2: NumNode): boolean {
  // We assume that the nodes are already compressed
  return node1 === node2;
}

type Relation = "LT" | "GT" | "EQ";

/**
 * Try to prove a relation between two nodes
 * This is a simplified version and can be expanded
 */
function canProveRelation(
  node1: NumNode,
  node2: NumNode,
  relation: Relation,
): boolean {
  // We can only compare certain types of nodes currently
  if (node1 instanceof LiteralNum && node2 instanceof LiteralNum) {
    switch (relation) {
      case "LT":
        return node1.value < node2.value;
      case "GT":
        return node1.value > node2.value;
      case "EQ":
        return node1.value === node2.value;
    }
  }

  // Add more sophisticated relation proofs here
  // For example, proving x < x+1 for any x

  return false;
}
