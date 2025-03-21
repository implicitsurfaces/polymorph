import { visitFromLeaves } from "../dag-tools/dag-traversal";
import {
  NumNode,
  childrenOfNumNode,
  LiteralNum,
  Variable,
  Derivative,
  UnaryOp,
  BinaryOp,
} from "../num-tree";
import { NumEvalKernel } from "../types";

export function genericEval<T>(root: NumNode, kernel: NumEvalKernel<T>): T {
  const evaledNodes = new Map<NumNode, T>();

  visitFromLeaves(root, childrenOfNumNode, (node) => {
    let evaled;
    if (node instanceof LiteralNum) {
      evaled = kernel.literal(node.value, node);
    } else if (node instanceof Variable) {
      evaled = kernel.variable(node.name, node);
    } else if (node instanceof Derivative) {
      const derivativeName = `d_${node.variable.name}`;
      evaled = kernel.variable(derivativeName, node);
    } else if (node instanceof UnaryOp) {
      const operand = evaledNodes.get(node.original)!;
      evaled = kernel.unaryOp(node.operation, operand, node);
    } else if (node instanceof BinaryOp) {
      const left = evaledNodes.get(node.left)!;
      const right = evaledNodes.get(node.right)!;

      evaled = kernel.binaryOp(node.operation, left, right, node);
    } else {
      throw new Error(
        `Unknown node type: ${node?.operation} ${node.constructor.name}`,
      );
    }
    evaledNodes.set(node, evaled);
  });

  return evaledNodes.get(root)!;
}
