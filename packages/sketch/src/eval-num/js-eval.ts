import { visitFromLeaves } from "../dag-tools/dag-traversal";
import { JSEvalKernel } from "./kernels/jsEval";
import {
  NumNode,
  LiteralNum,
  Variable,
  Derivative,
  UnaryOp,
  BinaryOp,
  childrenOfNumNode,
  BinaryOperation,
  UnaryOperation,
} from "../num-tree";
import { NumEvalKernel } from "../types";
import { genericEval } from "./genericEval";

export const naiveEval = (
  node: NumNode,
  variables: Map<string, number>,
): number => {
  const kernel = new JSEvalKernel(variables);
  return genericEval(node, kernel);
};

class NaNReportingKernel extends JSEvalKernel {
  private reportNaN(message: string) {
    //fs.writeFileSync("error.dot", renderNodeAsDot(treeEval(node)));
    throw new Error(message);
  }

  unaryOp(operation: UnaryOperation, operand: number) {
    const result = super.unaryOp(operation, operand);
    if (Number.isNaN(result)) {
      this.reportNaN(`NaN in unary op: ${operation}(${operand})`);
    }
    return result;
  }

  binaryOp(operation: BinaryOperation, lhs: number, rhs: number) {
    const result = super.binaryOp(operation, lhs, rhs);
    if (Number.isNaN(result)) {
      this.reportNaN(`NaN in binary op: ${lhs} ${operation} ${rhs}`);
    }
    return result;
  }
}

export const simpleEval = function (
  node: NumNode,
  variables: Map<string, number> = new Map(),
): number {
  const kernel = new NaNReportingKernel(variables);
  return genericEval(node, kernel);
};

export function treeEval<T = number>(
  root: NumNode,
  kernel: NumEvalKernel<T> = new JSEvalKernel() as unknown as NumEvalKernel<T>,
): NumNode & { evalsTo: number } {
  const outNodes = new Map<NumNode, NumNode & { evalsTo: number }>();
  const evaledNodes = new Map<NumNode, T>();

  visitFromLeaves(root, childrenOfNumNode, (node) => {
    let evaled;
    let extended: NumNode & { evalsTo: number };

    if (node instanceof LiteralNum) {
      evaled = kernel.literal(node.value, node);
      extended = Object.assign(new LiteralNum(node.value), {
        evalsTo: kernel.value(evaled),
      });
    } else if (node instanceof Variable) {
      evaled = kernel.variable(node.name, node);
      extended = Object.assign(new Variable(node.name), {
        evalsTo: kernel.value(evaled),
      });
    } else if (node instanceof Derivative) {
      const derivativeName = `d_${node.variable.name}`;
      evaled = kernel.variable(derivativeName, node);
      extended = Object.assign(new Derivative(node.variable), {
        evalsTo: kernel.value(evaled),
      });
    } else if (node instanceof UnaryOp) {
      const operand = evaledNodes.get(node.original)!;
      evaled = kernel.unaryOp(node.operation, operand, node);
      extended = Object.assign(
        new UnaryOp(node.operation, outNodes.get(node.original)!),
        { evalsTo: kernel.value(evaled) },
      );
    } else if (node instanceof BinaryOp) {
      const left = evaledNodes.get(node.left)!;
      const right = evaledNodes.get(node.right)!;

      evaled = kernel.binaryOp(node.operation, left, right, node);
      extended = Object.assign(
        new BinaryOp(
          node.operation,
          outNodes.get(node.left)!,
          outNodes.get(node.right)!,
        ),
        { evalsTo: kernel.value(evaled) },
      );
    } else {
      throw new Error(
        `Unknown node type: ${node?.operation} ${node.constructor.name}`,
      );
    }

    outNodes.set(node, extended);
    evaledNodes.set(node, evaled);
  });

  return outNodes.get(root)!;
}

export async function dedupeEval(
  node: NumNode,
  vars: Map<string, number> = new Map(),
): Promise<number> {
  /* this is actually way slower than simpleEval */
  /*
  const deduped = await dedupeTree(node);
  return simpleEval(deduped);
 
  to be kept for now
  */
  return Promise.resolve(simpleEval(node, vars));
}
