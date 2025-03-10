import { Context, Node as FidgetNode } from "fidget";

import { BinaryOperation, UnaryOperation } from "../../types";
import { NumEvalKernel } from "../../types";

export class FidgetEvalKernel implements NumEvalKernel<FidgetNode> {
  private varCache = new Map<string, FidgetNode>();
  constructor(
    private context: Context,
    private valuedVars: Map<string, number> = new Map(),
  ) {}

  variable(name: string): FidgetNode {
    if (this.valuedVars.has(name)) {
      return this.context.constant(this.valuedVars.get(name)!);
    }

    if (name === "x") {
      return this.context.x();
    } else if (name === "y") {
      return this.context.y();
    } else if (name === "z") {
      return this.context.z();
    }
    if (!this.varCache.has(name)) {
      this.varCache.set(name, this.context.var());
    }
    return this.varCache.get(name)!;
  }

  literal(value: number): FidgetNode {
    return this.context.constant(value);
  }

  value(value: FidgetNode): number {
    return this.context.evalNode(value);
  }

  unaryOp(operation: UnaryOperation, operand: FidgetNode): FidgetNode {
    if (operation === "SQRT") {
      return this.context.sqrt(operand);
    }
    if (operation === "COS") {
      return this.context.cos(operand);
    }
    if (operation === "ACOS") {
      return this.context.acos(operand);
    }
    if (operation === "ASIN") {
      return this.context.asin(operand);
    }
    if (operation === "TAN") {
      return this.context.tan(operand);
    }
    if (operation === "ATAN") {
      return this.context.atan(operand);
    }
    if (operation === "LOG") {
      return this.context.ln(operand);
    }
    if (operation === "EXP") {
      return this.context.exp(operand);
    }
    if (operation === "ABS") {
      return this.context.abs(operand);
    }
    if (operation === "NEG") {
      return this.context.neg(operand);
    }
    if (operation === "SIN") {
      return this.context.sin(operand);
    }
    if (operation === "NOT") {
      return this.context.not(operand);
    }
    if (operation === "SIGN") {
      return this.context.compare(operand, this.context.constant(0));
    }
    if (operation === "TANH") {
      // This should be implemented in fidget, using builtin operations
      const exp2x = this.context.exp(
        this.context.mul(this.context.constant(2), operand),
      );
      return this.context.div(
        this.context.sub(exp2x, this.context.constant(1)),
        this.context.add(exp2x, this.context.constant(1)),
      );
    }
    if (operation === "LOG1P") {
      // This should be implemented in fidget, using builtin operations
      return this.context.ln(
        this.context.add(operand, this.context.constant(1)),
      );
    }
    throw new Error(`Unknown unary operation: ${operation}`);
  }

  binaryOp(
    operation: BinaryOperation,
    left: FidgetNode,
    right: FidgetNode,
  ): FidgetNode {
    if (operation === "ADD") {
      return this.context.add(left, right);
    }
    if (operation === "SUB") {
      return this.context.sub(left, right);
    }
    if (operation === "MUL") {
      return this.context.mul(left, right);
    }
    if (operation === "DIV") {
      const r = this.context.add(right, this.context.constant(1e-30));
      return this.context.div(left, r);
    }
    if (operation === "MOD") {
      return this.context.modulo(left, right);
    }
    if (operation === "ATAN2") {
      return this.context.atan2(left, right);
    }
    if (operation === "MIN") {
      return this.context.min(left, right);
    }
    if (operation === "MAX") {
      return this.context.max(left, right);
    }
    if (operation === "COMPARE") {
      return this.context.compare(left, right);
    }
    if (operation === "AND") {
      return this.context.and(left, right);
    }
    if (operation === "OR") {
      return this.context.or(left, right);
    }
    throw new Error(`Unknown binary operation: ${operation}`);
  }
}
