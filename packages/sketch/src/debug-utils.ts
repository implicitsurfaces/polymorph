import type Decimal from "decimal.js";
import { genericEval } from "./eval-num/genericEval";
import { treeEval } from "./eval-num/js-eval";
import { JSEvalKernel } from "./eval-num/kernels/jsEval";
import { JSPrecisionEvalKernel } from "./eval-num/kernels/jsPrecisionEval";
import { Num, NumX, NumY } from "./num";

// @ts-expect-error -- no types
import { writeFileSync } from "node:fs";
import { renderNodeAsDot } from "./eval-num/dot-eval";
import { DistField } from "./types";
import { Point } from "./geom";

export function logDebug(
  num: Num,
  variables: Map<string, number>,
  enhancedPrecision = false,
) {
  const kernel = enhancedPrecision
    ? new JSPrecisionEvalKernel(variables, true)
    : new JSEvalKernel(variables, true);
  genericEval<number | Decimal>(num.n, kernel);
}

export function writeTreeAsDot(
  num: Num | NumNode,
  variables: Map<string, number> | Record<string, number>,
  enhancedPrecision = false,
  path = "tree.dot",
) {
  const vars =
    variables instanceof Map ? variables : new Map(Object.entries(variables));
  const kernel = enhancedPrecision
    ? new JSPrecisionEvalKernel(vars, true)
    : new JSEvalKernel(vars, true);

  const evaled = treeEval<number | Decimal>(
    num instanceof Num ? num.n : num,
    kernel,
  );
  writeFileSync(path, renderNodeAsDot(evaled));
}

export function pointVars(x: number, y: number) {
  return new Map([
    ["x", x],
    ["y", y],
  ]);
}

export function profile(p: DistField) {
  return p.distanceTo(new Point(NumX, NumY));
}
