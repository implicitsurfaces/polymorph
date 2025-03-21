import { genericEval } from "./eval-num/genericEval";
import { DiffEvalKernel } from "./eval-num/kernels/diffEval";
import { NumNode } from "./num-tree";

export function fullDerivative(node: NumNode): NumNode {
  return genericEval(node, new DiffEvalKernel());
}
