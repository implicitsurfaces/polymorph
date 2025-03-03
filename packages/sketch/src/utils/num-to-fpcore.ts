/**
 * Convert a NumNode expression tree to FPCore format
 */

import {
  NumNode,
  UnaryOp,
  BinaryOp,
  LiteralNum,
  Variable,
  Derivative,
  UnaryOperation,
  BinaryOperation,
} from "../num-tree";
import { memoizeNodeEval } from "./cache";

// Mapping of operation names from NumNode to FPCore
const unaryOpMap: Record<UnaryOperation, string> = {
  SQRT: "sqrt",
  COS: "cos",
  ACOS: "acos",
  ASIN: "asin",
  TAN: "tan",
  ATAN: "atan",
  LOG: "log",
  EXP: "exp",
  ABS: "fabs",
  NEG: "-",
  SIN: "sin",
  SIGN: "sign",
  NOT: "not",
  TANH: "tanh",
  LOG1P: "log1p",
};

const binaryOpMap: Record<BinaryOperation, string> = {
  ADD: "+",
  SUB: "-",
  MUL: "*",
  DIV: "/",
  MOD: "mod",
  ATAN2: "atan2",
  MIN: "fmin",
  MAX: "fmax",
  COMPARE: "compare",
  AND: "and",
  OR: "or",
};

/**
 * Convert a NumNode to its FPCore expression string
 */
const numNodeToFPCoreExpr = memoizeNodeEval(function numNodeToFPCoreExprInner(
  node: NumNode,
): string {
  if (node instanceof LiteralNum) {
    // Handle numeric literals
    return node.value.toString();
  } else if (node instanceof Variable) {
    // Handle variables
    return node.name;
  } else if (node instanceof UnaryOp) {
    const innerExpr = numNodeToFPCoreExpr(node.original);
    if (node.operation === "SIGN") {
      return `(if (> ${innerExpr} 0) 1 (if (< ${innerExpr} 0) -1 0))`;
    }
    if (node.operation === "NOT") {
      return `(if (== ${innerExpr} 0) 1 0)`;
    }
    // Handle unary operations
    const fpOp = unaryOpMap[node.operation];
    return `(${fpOp} ${innerExpr})`;
  } else if (node instanceof BinaryOp) {
    const leftExpr = numNodeToFPCoreExpr(node.left);
    const rightExpr = numNodeToFPCoreExpr(node.right);
    if (node.operation === "COMPARE") {
      // Handle COMPARE specially using if expressions
      // Return 1 if left > right, -1 if left < right, 0 if equal
      const innerExpr = `(- ${leftExpr} ${rightExpr})`;
      return `(if (> ${innerExpr} 0) 1 (if (< ${innerExpr} 0) -1 0))`;
    }
    if (node.operation === "AND") {
      // Handle COMPARE specially using if expressions
      // Return 1 if left > right, -1 if left < right, 0 if equal
      return `(if (== ${leftExpr} 0) ${leftExpr} ${rightExpr})`;
    }
    if (node.operation === "OR") {
      // Handle COMPARE specially using if expressions
      // Return 1 if left > right, -1 if left < right, 0 if equal
      return `(if (== ${leftExpr} 0) ${rightExpr} ${leftExpr})`;
    }

    const fpOp = binaryOpMap[node.operation];
    return `(${fpOp} ${leftExpr} ${rightExpr})`;
  } else if (node instanceof Derivative) {
    // This is a special case - FPCore doesn't have a built-in derivative operator
    // You might want to handle this differently based on your requirements
    throw new Error(
      "Derivative operations are not directly supported in FPCore",
    );
  } else {
    throw new Error(`Unknown node type: ${node.constructor.name}`);
  }
});

/**
 * Collect all unique variables from a NumNode tree
 */
function collectVariables(
  node: NumNode,
  variables: Set<string> = new Set<string>(),
  visited: Set<NumNode> = new Set<NumNode>(),
): Set<string> {
  if (visited.has(node)) {
    return variables;
  }
  if (node instanceof Variable) {
    variables.add(node.name);
  } else if (node instanceof UnaryOp) {
    collectVariables(node.original, variables, visited);
  } else if (node instanceof BinaryOp) {
    collectVariables(node.left, variables, visited);
    collectVariables(node.right, variables, visited);
  } else if (node instanceof Derivative) {
    if (node.variable instanceof Variable) {
      variables.add(node.variable.name);
    }
  }
  visited.add(node);
  return variables;
}

/**
 * Convert a NumNode expression tree to a complete FPCore definition
 *
 * @param node The root of the expression tree
 * @param name Optional name for the FPCore function
 * @param properties Optional properties for the FPCore function
 * @returns A string containing the FPCore definition
 */
export function numNodeToFPCore(
  node: NumNode,
  name?: string,
  properties: string[] = [],
): string {
  // Collect all variable names
  console.log("collecting variables");
  const variables = Array.from(collectVariables(node)).sort();

  // Build the inputs part
  const inputs = `(${variables.join(" ")})`;

  console.log("variables", inputs);

  console.log("building expression");

  // Build the expression part
  const expr = numNodeToFPCoreExpr(node);
  console.log("expr done");

  // Build the full FPCore
  const nameStr = name ? ` :name "${name}"` : "";
  const propsStr = properties.length > 0 ? " " + properties.join(" ") : "";
  return `(FPCore ${inputs}${nameStr}${propsStr} ${expr})`;
}

// Example usage:
// const expr = new BinaryOp(
//   "ADD",
//   new BinaryOp("MUL", new Variable("a"), new Variable("a")),
//   new BinaryOp("MUL", new Variable("b"), new Variable("b"))
// );
// const result = numNodeToFPCore(expr, "hypotenuse");
// console.log(result);
// Output: (FPCore hypotenuse (a b) (+ (* a a) (* b b)))
