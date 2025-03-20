import { createContext } from "fidget";

import { BinaryOp, LiteralNum, NumNode, Variable, UnaryOp } from "../num-tree";
import { DistField, SolidDistField } from "../types";
import { vecFromCartesianCoords } from "../geom";
import { NumX, NumY, NumZ, ZERO } from "../num";
import { vec3FromCartesianCoords } from "../geom-3d";
import { genericEval } from "./genericEval";
import { FidgetEvalKernel } from "./kernels/fidgetEval";

export async function fidgetEval(node: NumNode): Promise<number> {
  const context = await createContext();
  const kernel = new FidgetEvalKernel(context);

  const fidgetNode = genericEval(node, kernel);
  return context.evalNode(fidgetNode);
}

export async function fidgetRender(
  node: DistField,
  imageSize = 50,
  colorPlot = false,
  valuedVars: Map<string, number> = new Map(),
): Promise<Uint8Array> {
  const context = await createContext();

  const genericPoint = vecFromCartesianCoords(NumX, NumY).pointFromOrigin();
  const kernel = new FidgetEvalKernel(context, valuedVars);
  const fidgetNode = genericEval(node.distanceTo(genericPoint).n, kernel);
  const render = context.renderNode(fidgetNode, imageSize, colorPlot);
  return render;
}

export async function fidgetRenderNode3D(
  node: SolidDistField,
  imageSize = 50,
  useHeightmap = false,
  valuedVars: Map<string, number> = new Map(),
): Promise<Uint8Array> {
  const context = await createContext();

  const genericPoint = vec3FromCartesianCoords(
    NumX,
    NumY,
    NumZ,
  ).pointFromOrigin();

  const kernel = new FidgetEvalKernel(context, valuedVars);
  const fidgetNode = genericEval(node.valueAt(genericPoint).n, kernel);

  const render = context.renderNodeIn3D(fidgetNode, imageSize, useHeightmap);
  return render;
}

const OP_MAP: Record<string, string> = {
  LOG: "ln",
};

const mapOperationName = (op: string) => {
  return OP_MAP[op] || op.toLowerCase();
};

export function nodeToString(
  node: NumNode,
  opList: string[],
  cache: Map<NumNode, string>,
): string {
  if (cache.has(node)) {
    return cache.get(node)!;
  }

  const out = (s: string) => {
    const index = opList.length;
    const id = `_${index.toString(16)}`;
    const op = `${id} ${s}`;
    opList.push(op);
    cache.set(node, id);
    return id;
  };

  if (node instanceof LiteralNum) {
    return out(`const ${node.value}`);
  } else if (node instanceof Variable) {
    if (node.name === "x") {
      return out("var-x");
    } else if (node.name === "y") {
      return out("var-y");
    } else if (node.name === "z") {
      return out("var-z");
    }
    throw new Error(`Unknown variable: ${node.name}`);
  } else if (node instanceof UnaryOp) {
    const operand = nodeToString(node.original, opList, cache);
    if (node.operation === "DEBUG") return operand;

    if (node.operation === "SIGN") {
      const zero = nodeToString(ZERO.n, opList, cache);
      return out(`compare ${operand} ${zero}`);
    }

    if (node.operation === "CBRT") {
      const eps = nodeToString(new LiteralNum(3e-16), opList, cache);
      const zero = nodeToString(ZERO.n, opList, cache);
      const three = nodeToString(new LiteralNum(3), opList, cache);
      const sign = out(`compare ${operand} ${zero}`);

      const abs = out(`abs ${operand}`);
      const m = out(`add ${abs} ${eps}`);
      const ln = out(`ln ${m}`);
      const lnDiv3 = out(`div ${ln} ${three}`);
      const exp = out(`exp ${lnDiv3}`);

      return out(`mul ${exp} ${sign}`);
    }

    return out(`${mapOperationName(node.operation)} ${operand}`);
  } else if (node instanceof BinaryOp) {
    const left = nodeToString(node.left, opList, cache);
    const right = nodeToString(node.right, opList, cache);

    return out(`${mapOperationName(node.operation)} ${left} ${right}`);
  }

  throw new Error(`Unknown node type: ${node?.operation}`);
}

export function fidgetStringify(node: NumNode): string {
  const opList: string[] = [];
  const cache = new Map<NumNode, string>();
  nodeToString(node, opList, cache);
  return opList.join("\n");
}
