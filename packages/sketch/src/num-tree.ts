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
  | "NOT";

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

export class NumNode {}

export class UnaryOp extends NumNode {
  constructor(
    public operation: UnaryOperation,
    public original: NumNode,
  ) {
    super();
  }
}

export class BinaryOp extends NumNode {
  constructor(
    public operation: BinaryOperation,
    public left: NumNode,
    public right: NumNode,
  ) {
    super();
  }
}

export class LiteralNum extends NumNode {
  constructor(public value: number) {
    super();
  }
}

export function simple_eval(node: NumNode): number {
  if (node instanceof LiteralNum) {
    return node.value;
  } else if (node instanceof UnaryOp) {
    const operand = simple_eval(node.original);

    if (node.operation === "SQRT") {
      return Math.sqrt(operand);
    }
    if (node.operation === "COS") {
      return Math.cos(operand);
    }
    if (node.operation === "ACOS") {
      return Math.acos(operand);
    }
    if (node.operation === "ASIN") {
      return Math.asin(operand);
    }
    if (node.operation === "TAN") {
      return Math.tan(operand);
    }
    if (node.operation === "ATAN") {
      return Math.atan(operand);
    }
    if (node.operation === "LOG") {
      return Math.log(operand);
    }
    if (node.operation === "EXP") {
      return Math.exp(operand);
    }
    if (node.operation === "ABS") {
      return Math.abs(operand);
    }
    if (node.operation === "NEG") {
      return -operand;
    }
    if (node.operation === "SIN") {
      return Math.sin(operand);
    }
    if (node.operation === "SIGN") {
      return Math.sign(operand);
    }
    if (node.operation === "NOT") {
      return operand ? 0 : 1;
    }
  } else if (node instanceof BinaryOp) {
    const left = simple_eval(node.left);
    const right = simple_eval(node.right);

    if (node.operation === "ADD") {
      return left + right;
    }
    if (node.operation === "SUB") {
      return left - right;
    }
    if (node.operation === "MUL") {
      return left * right;
    }
    if (node.operation === "DIV") {
      return left / right;
    }
    if (node.operation === "ATAN2") {
      return Math.atan2(left, right);
    }
    if (node.operation === "MOD") {
      return left % right;
    }
    if (node.operation === "MIN") {
      return Math.min(left, right);
    }
    if (node.operation === "MAX") {
      return Math.max(left, right);
    }
    if (node.operation === "COMPARE") {
      return Math.sign(left - right);
    }
    if (node.operation === "AND") {
      return left && right;
    }
    if (node.operation === "OR") {
      return left || right;
    }
  }
}
