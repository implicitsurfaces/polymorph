import { vecFromCartesianCoords } from "./geom";
import { vec3FromCartesianCoords } from "./geom-3d";
import { NumX, NumY, NumZ } from "./num";
import { allVariables } from "./num-tree";
import { naiveEval } from "./eval-num/js-eval";
import { evalSketch } from "./eval-sketch/eval";
import {
  fidgetRender,
  fidgetRenderNode3D,
  fidgetStringify,
} from "./eval-num/fidget-eval";
import { gradientDescentOpt } from "./opt";
import { AnyProfileNode, AnySolidNode } from "./sketch-nodes/types";
import {
  AnyAngleNode,
  AnyConstraintNode,
  AnyDistanceNode,
  AnyPoint3Node,
  AnyPointNode,
  AnyRealValueNode,
  AnyVector3Node,
  AnyVectorNode,
} from "./sketch-nodes";
import { isDistField } from "./types";

export async function renderProfile(
  profile: AnyProfileNode,
  valuedVars: Map<string, number> = new Map(),
  size: number,
) {
  const distField = evalSketch(profile);
  const render = await fidgetRender(distField, size, true, valuedVars);
  return new Uint8ClampedArray(render);
}

export async function renderSolid(
  solid: AnySolidNode,
  valuedVars: Map<string, number> = new Map(),
  size: number,
) {
  const distField = evalSketch(solid);
  const render = await fidgetRenderNode3D(distField, size, true, valuedVars);
  return new Uint8ClampedArray(render);
}

export async function debugRenderProfile(
  profile: AnyProfileNode,
  valuedVars: Map<string, number> = new Map(),
  size = 50,
) {
  const distField = evalSketch(profile);
  return await fidgetRender(distField, size, false, valuedVars);
}

export function readDistance(
  distance: AnyDistanceNode,
  valuedVars: Map<string, number>,
) {
  const d = evalSketch(distance);
  return naiveEval(d.n, valuedVars);
}

export function readRealValue(
  distance: AnyRealValueNode,
  valuedVars: Map<string, number>,
) {
  if (typeof distance === "number") {
    return distance;
  }
  const d = evalSketch(distance);
  return naiveEval(d.n, valuedVars);
}

export function readAngleAsDegree(
  angle: AnyAngleNode,
  valuedVars: Map<string, number>,
) {
  const d = evalSketch(angle);
  return naiveEval(d.asDeg().n, valuedVars);
}

export function readPoint(
  point: AnyPointNode,
  valuedVars: Map<string, number>,
): [number, number] {
  const p = evalSketch(point);
  return [naiveEval(p.x.n, valuedVars), naiveEval(p.y.n, valuedVars)];
}

export function readVector(
  vector: AnyVectorNode,
  valuedVars: Map<string, number>,
): [number, number] {
  const v = evalSketch(vector);
  return [naiveEval(v.x.n, valuedVars), naiveEval(v.y.n, valuedVars)];
}

export function readPoint3(
  point: AnyPoint3Node,
  valuedVars: Map<string, number>,
): [number, number, number] {
  const p = evalSketch(point);
  return [
    naiveEval(p.x.n, valuedVars),
    naiveEval(p.y.n, valuedVars),
    naiveEval(p.z.n, valuedVars),
  ];
}

export function readVector3(
  vector: AnyVector3Node,
  valuedVars: Map<string, number>,
): [number, number, number] {
  const v = evalSketch(vector);
  return [
    naiveEval(v.x.n, valuedVars),
    naiveEval(v.y.n, valuedVars),
    naiveEval(v.z.n, valuedVars),
  ];
}

export function findSolution(
  constraints: AnyConstraintNode[],
  options: {
    learningRate?: number;
    maxSteps?: number;
    tolerance?: number;
    momentum?: number;
    debug?: boolean;
  } = {},
) {
  if (constraints.length === 0) {
    return { solution: new Map<string, number>(), change: 0, steps: 0 };
  }
  let loss = evalSketch(constraints[0]);

  constraints.slice(1).forEach((term) => {
    loss = loss.add(evalSketch(term));
  });

  const vars = allVariables(loss.n);
  if (vars.size === 0) {
    return { solution: new Map<string, number>(), change: 0, steps: 0 };
  }

  const initialX = new Map<string, number>(
    [...vars.keys()].map((key) => [key, 0.0]),
  );

  return gradientDescentOpt(loss, initialX, options);
}

export function exportAsFidget(node: AnySolidNode | AnyProfileNode) {
  const generic2dPoint = vecFromCartesianCoords(NumX, NumY).pointFromOrigin();
  const generic3dPoint = vec3FromCartesianCoords(
    NumX,
    NumY,
    NumZ,
  ).pointFromOrigin();

  const evaled = evalSketch(node);

  const distNode = isDistField(evaled)
    ? evaled.distanceTo(generic2dPoint)
    : evaled.valueAt(generic3dPoint);
  return fidgetStringify(distNode.n);
}
