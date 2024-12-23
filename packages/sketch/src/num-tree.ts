import { memoizeNodeEval } from "./utils/cache";
//import { renderNodeAsDot } from "./utils/num-to-dot";
//import fs from "node:fs";

export type UnaryOperation =
  | "SQRT"
  | "COS"
  | "ACOS"
  | "ASIN"
  | "TAN"
  | "ATAN"
  | "LOG"
  | "EXP"
  | "ABS"
  | "NEG"
  | "SIN"
  | "SIGN"
  | "NOT"
  | "TANH"
  | "LOG1P";

export type BinaryOperation =
  | "ADD"
  | "SUB"
  | "MUL"
  | "DIV"
  | "MOD"
  | "ATAN2"
  | "MIN"
  | "MAX"
  | "COMPARE"
  | "AND"
  | "OR";

export class NumNode {
  readonly operation: string = "NONE";
}

export class Derivative extends NumNode {
  constructor(readonly variable: Variable) {
    super();
  }
}

export class UnaryOp extends NumNode {
  constructor(
    readonly operation: UnaryOperation,
    readonly original: NumNode,
  ) {
    super();
  }
}

export class BinaryOp extends NumNode {
  constructor(
    readonly operation: BinaryOperation,
    readonly left: NumNode,
    readonly right: NumNode,
  ) {
    super();
  }
}

export class LiteralNum extends NumNode {
  readonly operation = "LITERAL";
  constructor(public value: number) {
    super();
  }
}

export class Variable extends NumNode {
  readonly operation = "VAR";
  constructor(public name: string) {
    super();
  }
}

const simpleUnaryOp = (operation: UnaryOperation, operand: number) => {
  if (operation === "SQRT") {
    return Math.sqrt(operand);
  }
  if (operation === "COS") {
    return Math.cos(operand);
  }
  if (operation === "ACOS") {
    return Math.acos(operand);
  }
  if (operation === "ASIN") {
    return Math.asin(operand);
  }
  if (operation === "TAN") {
    return Math.tan(operand);
  }
  if (operation === "ATAN") {
    return Math.atan(operand);
  }
  if (operation === "LOG") {
    return Math.log(operand);
  }
  if (operation === "EXP") {
    return Math.exp(operand);
  }
  if (operation === "ABS") {
    return Math.abs(operand);
  }
  if (operation === "NEG") {
    return -operand;
  }
  if (operation === "SIN") {
    return Math.sin(operand);
  }
  if (operation === "SIGN") {
    return Math.sign(operand);
  }
  if (operation === "NOT") {
    return operand ? 0 : 1;
  }
  if (operation === "TANH") {
    return Math.tanh(operand);
  }
  if (operation === "LOG1P") {
    return Math.log1p(operand);
  }
  throw new Error(`Unknown unary operation: ${operation}`);
};

const simpleBinaryOp = (
  operation: BinaryOperation,
  left: number,
  right: number,
) => {
  if (operation === "ADD") {
    return left + right;
  }
  if (operation === "SUB") {
    return left - right;
  }
  if (operation === "MUL") {
    return left * right;
  }
  if (operation === "DIV") {
    return right ? left / right : 1e50;
  }
  if (operation === "MOD") {
    return left % right;
  }
  if (operation === "ATAN2") {
    return Math.atan2(left, right);
  }
  if (operation === "MIN") {
    return Math.min(left, right);
  }
  if (operation === "MAX") {
    return Math.max(left, right);
  }
  if (operation === "COMPARE") {
    return Math.sign(left - right);
  }
  if (operation === "AND") {
    return left && right;
  }
  if (operation === "OR") {
    return left || right;
  }
  throw new Error(`Unknown binary operation: ${operation}`);
};

export const ZERO_NODE = new LiteralNum(0);
export const ONE_NODE = new LiteralNum(1);
export const TWO_NODE = new LiteralNum(2);
export const NEG_ONE_NODE = new LiteralNum(-1);

export const naiveEval = (
  node: NumNode,
  variables: Map<string, number>,
): number => {
  if (node instanceof LiteralNum) {
    return node.value;
  } else if (node instanceof Variable) {
    const value = variables.get(node.name);
    if (value === undefined) {
      throw new Error(`Variable not found: ${node.name}`);
    }
    return value;
  } else if (node instanceof Derivative) {
    const derivativeName = `d_${node.variable.name}`;
    const value = variables.get(derivativeName);
    if (value === undefined) {
      throw new Error(`Variable not found: ${derivativeName}`);
    }
    return value;
  } else if (node instanceof UnaryOp) {
    const operand = naiveEval(node.original, variables);
    return simpleUnaryOp(node.operation, operand);
  } else if (node instanceof BinaryOp) {
    const left = naiveEval(node.left, variables);
    const right = naiveEval(node.right, variables);

    return simpleBinaryOp(node.operation, left, right);
  }

  throw new Error(
    `Unknown node type: ${node?.operation} ${node.constructor.name}`,
  );
};

export const replaceVariable = (
  node: NumNode,
  variables: Map<string, number | NumNode>,
): NumNode => {
  if (node instanceof Variable) {
    const value = variables.get(node.name);
    if (value === undefined) {
      return node;
    }
    if (value instanceof NumNode) {
      return value;
    }
    return new LiteralNum(value);
  } else if (node instanceof UnaryOp) {
    return new UnaryOp(
      node.operation,
      replaceVariable(node.original, variables),
    );
  } else if (node instanceof BinaryOp) {
    return new BinaryOp(
      node.operation,
      replaceVariable(node.left, variables),
      replaceVariable(node.right, variables),
    );
  }
  return node;
};

export const partialDerivative = (node: NumNode, variable: string): NumNode => {
  if (node instanceof Derivative) {
    if (node.variable.name === variable) {
      return ONE_NODE;
    }
    return ZERO_NODE;
  } else if (node instanceof UnaryOp) {
    const operand = partialDerivative(node.original, variable);
    return new UnaryOp(node.operation, operand);
  } else if (node instanceof BinaryOp) {
    const left = partialDerivative(node.left, variable);
    const right = partialDerivative(node.right, variable);
    return new BinaryOp(node.operation, left, right);
  }
  return node;
};

const reportNaN = (node: NumNode) => {
  //fs.writeFileSync("error.dot", renderNodeAsDot(treeEval(node)));
  throw new Error(`NaN in binary op: ${node.operation}`);
};

const simpleEval = memoizeNodeEval(function (node: NumNode): number {
  if (node instanceof LiteralNum) {
    return node.value;
  } else if (node instanceof UnaryOp) {
    const operand = simpleEval(node.original);
    if (Number.isNaN(operand)) {
      reportNaN(node.original);
    }
    return simpleUnaryOp(node.operation, operand);
  } else if (node instanceof BinaryOp) {
    const left = simpleEval(node.left);
    if (Number.isNaN(left)) {
      reportNaN(node.left);
    }
    const right = simpleEval(node.right);
    if (Number.isNaN(left)) {
      reportNaN(node.right);
    }

    return simpleBinaryOp(node.operation, left, right);
  }

  throw new Error(`Unknown node type: ${node?.operation}`);
});

export const treeEval = (node: NumNode): NumNode & { value: number } => {
  if (node instanceof LiteralNum) {
    return Object.assign(node, { value: node.value });
  } else if (node instanceof UnaryOp) {
    const operand = treeEval(node.original);
    const value = simpleUnaryOp(node.operation, operand.value);
    return Object.assign(node, {
      operand,
      value,
    });
  } else if (node instanceof BinaryOp) {
    const left = treeEval(node.left);
    const right = treeEval(node.right);

    const value = simpleBinaryOp(node.operation, left.value, right.value);
    return Object.assign(node, {
      left,
      right,
      value,
    });
  }

  throw new Error(`Unknown node type: ${node?.operation}`);
};

export async function dedupeEval(node: NumNode): Promise<number> {
  /* this is actually way slower than simpleEval */
  /*
  const deduped = await dedupeTree(node);
  return simpleEval(deduped);

  to be kept for now
  */

  return Promise.resolve(simpleEval(node));
}

const allVariables = memoizeNodeEval(function (node: NumNode): Set<string> {
  if (node instanceof Variable) {
    return new Set([node.name]);
  } else if (node instanceof UnaryOp) {
    return allVariables(node.original);
  } else if (node instanceof BinaryOp) {
    const left = allVariables(node.left);
    const right = allVariables(node.right);
    return new Set([...left, ...right]);
  }
  return new Set();
});

export { simpleEval, allVariables };
