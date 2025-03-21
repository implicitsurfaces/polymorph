import { DotEvalKernel } from "../eval-num/kernels/dotEval";
import { genericEval } from "../eval-num/genericEval";
import { NumNode } from "../num-tree";

export function renderNodeAsDot(root: NumNode & { evalsTo?: number }): string {
  const kernel = new DotEvalKernel();
  genericEval(root, kernel);
  return kernel.getDot();
}
