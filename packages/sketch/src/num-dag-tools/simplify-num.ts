import { genericEval } from "../eval-num/genericEval";
import { ConstantsFoldEval } from "../eval-num/kernels/foldConstants";
import type { NumNode } from "../num-tree";
import { compressNum } from "./compress-num";

export function simplify(node: NumNode): NumNode {
  return genericEval(compressNum(node), new ConstantsFoldEval());
}
