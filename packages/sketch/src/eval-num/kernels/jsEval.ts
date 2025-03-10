import type { BinaryOperation, UnaryOperation } from "../../types";
import { NumEvalKernel } from "../../types";

export class JSEvalKernel implements NumEvalKernel<number> {
  constructor(
    public readonly variablesValues: Map<string, number> = new Map(),
  ) {}
  value(value: number) {
    return value;
  }
  literal(value: number) {
    return value;
  }
  variable(name: string) {
    if (!this.variablesValues.has(name)) {
      throw new Error(`Unknown variable: ${name}`);
    }
    return this.variablesValues.get(name)!;
  }
  unaryOp(operation: UnaryOperation, operand: number) {
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
    if (operation === "DEBUG") {
      return operand;
    }
    throw new Error(`Unknown unary operation: ${operation}`);
  }

  binaryOp(operation: BinaryOperation, left: number, right: number) {
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
      return left === 0 ? left : right;
    }
    if (operation === "OR") {
      return left === 0 ? right : left;
    }
    throw new Error(`Unknown binary operation: ${operation}`);
  }
}
