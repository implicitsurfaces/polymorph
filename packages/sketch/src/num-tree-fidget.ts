import { Context, Node as FidgetNode, createContext } from "fidget";

import {
  BinaryOp,
  BinaryOperation,
  LiteralNum,
  NumNode,
  Variable,
  UnaryOp,
  UnaryOperation,
} from "./num-tree";
import { DistField } from "./types";
import { vecFromCartesianCoords } from "./geom";
import { NumX, NumY } from "./num";

export function _fidgetEval(node: NumNode, context: Context): FidgetNode {
  if (node instanceof LiteralNum) {
    return context.constant(node.value);
  } else if (node instanceof Variable) {
    if (node.name === "x") {
      return context.x();
    } else if (node.name === "y") {
      return context.y();
    } else if (node.name === "z") {
      return context.z();
    }
    return context.var();
  } else if (node instanceof UnaryOp) {
    const operand = _fidgetEval(node.original, context);
    return fidgetUnaryOp(node.operation, operand, context);
  } else if (node instanceof BinaryOp) {
    const left = _fidgetEval(node.left, context);
    const right = _fidgetEval(node.right, context);

    return fidgetBinaryOp(node.operation, left, right, context);
  }

  throw new Error(`Unknown node type: ${node?.operation}`);
}

function fidgetUnaryOp(
  operation: UnaryOperation,
  operand: number,
  context: Context,
): FidgetNode {
  if (operation === "SQRT") {
    return context.sqrt(operand);
  }
  if (operation === "COS") {
    return context.cos(operand);
  }
  if (operation === "ACOS") {
    return context.acos(operand);
  }
  if (operation === "ASIN") {
    return context.asin(operand);
  }
  if (operation === "TAN") {
    return context.tan(operand);
  }
  if (operation === "ATAN") {
    return context.atan(operand);
  }
  if (operation === "LOG") {
    return context.ln(operand);
  }
  if (operation === "EXP") {
    return context.exp(operand);
  }
  if (operation === "ABS") {
    return context.abs(operand);
  }
  if (operation === "NEG") {
    return context.neg(operand);
  }
  if (operation === "SIN") {
    return context.sin(operand);
  }
  if (operation === "NOT") {
    return context.not(operand);
  }
  if (operation === "SIGN") {
    return context.compare(operand, context.constant(0));
  }
  if (operation === "TANH") {
    // This should be implemented in fidget, using builtin operations
    const exp2x = context.exp(context.mul(context.constant(2), operand));
    return context.div(
      context.sub(exp2x, context.constant(1)),
      context.add(exp2x, context.constant(1)),
    );
  }
  if (operation === "LOG1P") {
    // This should be implemented in fidget, using builtin operations
    return context.ln(context.add(operand, context.constant(1)));
  }
  throw new Error(`Unknown unary operation: ${operation}`);
}

const fidgetBinaryOp = (
  operation: BinaryOperation,
  left: FidgetNode,
  right: FidgetNode,
  context: Context,
) => {
  if (operation === "ADD") {
    return context.add(left, right);
  }
  if (operation === "SUB") {
    return context.sub(left, right);
  }
  if (operation === "MUL") {
    return context.mul(left, right);
  }
  if (operation === "DIV") {
    return context.div(left, right);
  }
  if (operation === "MOD") {
    return context.modulo(left, right);
  }
  if (operation === "ATAN2") {
    return context.atan2(left, right);
  }
  if (operation === "MIN") {
    return context.min(left, right);
  }
  if (operation === "MAX") {
    return context.max(left, right);
  }
  if (operation === "COMPARE") {
    return context.compare(left, right);
  }
  if (operation === "AND") {
    return context.and(left, right);
  }
  if (operation === "OR") {
    return context.or(left, right);
  }
  throw new Error(`Unknown binary operation: ${operation}`);
};

export async function fidgetEval(node: NumNode): Promise<number> {
  const context = await createContext();
  const fidgetNode = _fidgetEval(node, context);
  return context.evalNode(fidgetNode);
}

export async function fidgetRender(
  node: DistField,
  imageSize = 50,
): Promise<Uint8Array> {
  const context = await createContext();
  const genericPoint = vecFromCartesianCoords(NumX, NumY).pointFromOrigin();
  const fidgetNode = _fidgetEval(node.distanceTo(genericPoint).n, context);
  return context.renderNode(fidgetNode, imageSize);
}
