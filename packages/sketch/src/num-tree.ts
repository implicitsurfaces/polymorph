import { memoizeNodeEval } from "./utils/cache";
import { dedupeTree } from "./utils/dedupe-tree";

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
    return left / right;
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

const simpleEval = memoizeNodeEval(function (node: NumNode): number {
  if (node instanceof LiteralNum) {
    return node.value;
  } else if (node instanceof UnaryOp) {
    const operand = simpleEval(node.original);
    return simpleUnaryOp(node.operation, operand);
  } else if (node instanceof BinaryOp) {
    const left = simpleEval(node.left);
    const right = simpleEval(node.right);

    return simpleBinaryOp(node.operation, left, right);
  }

  throw new Error(`Unknown node type: ${node?.operation}`);
});

export async function dedupeEval(node: NumNode): Promise<number> {
  /* this is actually way slower than simpleEval */
  /*
  const deduped = await dedupeTree(node);
  return simpleEval(deduped);

  to be kept for now
  */

  return Promise.resolve(simpleEval(node));
}

export { simpleEval };
