import {
  NumNode,
  UnaryOp,
  BinaryOp,
  LiteralNum,
  Variable,
  Derivative,
  ZERO_NODE as ZERO,
  ONE_NODE as ONE,
  DebugNode,
} from "../num-tree"; // Assuming the types are defined in a separate file
import { UnaryOperation, BinaryOperation } from "../types";
import { memoizeNodeEval } from "../utils/cache";

/**
 * Main function to simplify a numerical expression tree
 */
export const simplify = memoizeNodeEval(function simplifyInner(
  node: NumNode,
): NumNode {
  // Base cases
  if (
    node instanceof LiteralNum ||
    node instanceof Variable ||
    node instanceof Derivative
  ) {
    return node;
  }

  if (node instanceof UnaryOp) {
    return simplifyUnaryOp(node);
  }

  if (node instanceof BinaryOp) {
    return simplifyBinaryOp(node);
  }

  return node;
});

/**
 * Simplify unary operations
 */
function simplifyUnaryOp(node: UnaryOp): NumNode {
  // First simplify the operand
  const simplifiedOperand = simplify(node.original);

  // If operand is a literal, we can evaluate the operation
  if (simplifiedOperand instanceof LiteralNum && node.operation !== "DEBUG") {
    return evaluateUnaryOp(node.operation, simplifiedOperand.value);
  }

  // Identity and simplification rules
  switch (node.operation) {
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
  if (simplifiedOperand !== node.original) {
    if (node instanceof DebugNode)
      return new DebugNode(simplifiedOperand, node.debug);
    return new UnaryOp(node.operation, simplifiedOperand);
  }

  return node;
}

/**
 * Simplify binary operations
 */
function simplifyBinaryOp(node: BinaryOp): NumNode {
  // First simplify both operands
  const simplifiedLeft = simplify(node.left);
  const simplifiedRight = simplify(node.right);

  // If both operands are literals, we can evaluate the operation
  if (
    simplifiedLeft instanceof LiteralNum &&
    simplifiedRight instanceof LiteralNum
  ) {
    return evaluateBinaryOp(
      node.operation,
      simplifiedLeft.value,
      simplifiedRight.value,
    );
  }

  // Handle specific cases for optimization
  switch (node.operation) {
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
      if (simplifiedLeft instanceof LiteralNum && simplifiedLeft.value !== 0) {
        return simplifiedRight;
      }
      break;

    case "OR":
      // true OR x = true
      if (simplifiedLeft instanceof LiteralNum && simplifiedLeft.value !== 0) {
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
  if (simplifiedLeft !== node.left || simplifiedRight !== node.right) {
    return new BinaryOp(node.operation, simplifiedLeft, simplifiedRight);
  }

  return node;
}

/**
 * Evaluate a unary operation on a literal value
 */
function evaluateUnaryOp(operation: UnaryOperation, value: number): LiteralNum {
  let result: number;

  switch (operation) {
    case "SQRT":
      result = Math.sqrt(value);
      break;
    case "CBRT":
      result = Math.cbrt(value);
      break;
    case "COS":
      result = Math.cos(value);
      break;
    case "SIN":
      result = Math.sin(value);
      break;
    case "ACOS":
      result = Math.acos(value);
      break;
    case "ASIN":
      result = Math.asin(value);
      break;
    case "TAN":
      result = Math.tan(value);
      break;
    case "ATAN":
      result = Math.atan(value);
      break;
    case "LOG":
      result = Math.log(value);
      break;
    case "EXP":
      result = Math.exp(value);
      break;
    case "ABS":
      result = Math.abs(value);
      break;
    case "NEG":
      result = -value;
      break;
    case "SIGN":
      result = Math.sign(value);
      break;
    case "NOT":
      result = value ? 0 : 1; // Logical NOT
      break;
    case "TANH":
      result = Math.tanh(value);
      break;
    case "LOG1P":
      result = Math.log1p(value);
      break;
    default:
      throw new Error(`Unknown unary operation: ${operation}`);
  }

  return new LiteralNum(result);
}

/**
 * Evaluate a binary operation on two literal values
 */
function evaluateBinaryOp(
  operation: BinaryOperation,
  left: number,
  right: number,
): LiteralNum {
  let result: number;

  switch (operation) {
    case "ADD":
      result = left + right;
      break;
    case "SUB":
      result = left - right;
      break;
    case "MUL":
      result = left * right;
      break;
    case "DIV":
      result = left / right;
      break;
    case "MOD":
      result = left % right;
      break;
    case "ATAN2":
      result = Math.atan2(left, right);
      break;
    case "MIN":
      result = Math.min(left, right);
      break;
    case "MAX":
      result = Math.max(left, right);
      break;
    case "COMPARE":
      result = left === right ? 0 : left < right ? -1 : 1;
      break;
    case "AND":
      result = left && right;
      break;
    case "OR":
      result = left || right;
      break;
    default:
      throw new Error(`Unknown binary operation: ${operation}`);
  }

  return new LiteralNum(result);
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
