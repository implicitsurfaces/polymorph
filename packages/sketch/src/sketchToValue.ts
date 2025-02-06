import { vecFromCartesianCoords } from "./geom";
import { vec3FromCartesianCoords } from "./geom-3d";
import { NumX, NumY, NumZ } from "./num";
import { allVariables, naiveEval } from "./num-tree";
import {
  fidgetRender,
  fidgetRenderNode3D,
  fidgetStringify,
} from "./num-tree-fidget";
import { gradientDescentOpt } from "./opt";
import {
  AngleNode,
  ConstraintNode,
  DistanceNode,
  Point3Node,
  PointNode,
  ProfileNode,
  RealValueNode,
  SolidNode,
  Vector3Node,
  VectorNode,
} from "./sketch-nodes";
import {
  evalAngle,
  evalConstraint,
  evalDistance,
  evalPoint,
  evalPoint3,
  evalProfile,
  evalRealValue,
  evalSolid,
  evalVector,
  evalVector3,
} from "./sketch-tree";

export async function renderProfile(
  profile: ProfileNode,
  valuedVars: Map<string, number> = new Map(),
  size: number,
) {
  const distField = evalProfile(profile);
  const render = await fidgetRender(distField, size, true, valuedVars);
  return new Uint8ClampedArray(render);
}

export async function renderSolid(
  solid: SolidNode,
  valuedVars: Map<string, number> = new Map(),
  size: number,
) {
  const distField = evalSolid(solid);
  const render = await fidgetRenderNode3D(distField, size, true, valuedVars);
  return new Uint8ClampedArray(render);
}

export async function debugRenderProfile(
  profile: ProfileNode,
  valuedVars: Map<string, number> = new Map(),
  size = 50,
) {
  const distField = evalProfile(profile);
  return await fidgetRender(distField, size, false, valuedVars);
}

export function readDistance(
  distance: DistanceNode,
  valuedVars: Map<string, number>,
) {
  const d = evalDistance(distance);
  return naiveEval(d.n, valuedVars);
}

export function readRealValue(
  distance: RealValueNode,
  valuedVars: Map<string, number>,
) {
  const d = evalRealValue(distance);
  return naiveEval(d.n, valuedVars);
}

export function readAngleAsDegree(
  angle: AngleNode,
  valuedVars: Map<string, number>,
) {
  const d = evalAngle(angle);
  return naiveEval(d.asDeg().n, valuedVars);
}

export function readPoint(
  point: PointNode,
  valuedVars: Map<string, number>,
): [number, number] {
  const p = evalPoint(point);
  return [naiveEval(p.x.n, valuedVars), naiveEval(p.y.n, valuedVars)];
}

export function readVector(
  vector: VectorNode,
  valuedVars: Map<string, number>,
): [number, number] {
  const v = evalVector(vector);
  return [naiveEval(v.x.n, valuedVars), naiveEval(v.y.n, valuedVars)];
}

export function readPoint3(
  point: Point3Node,
  valuedVars: Map<string, number>,
): [number, number, number] {
  const p = evalPoint3(point);
  return [
    naiveEval(p.x.n, valuedVars),
    naiveEval(p.y.n, valuedVars),
    naiveEval(p.z.n, valuedVars),
  ];
}

export function readVector3(
  vector: Vector3Node,
  valuedVars: Map<string, number>,
): [number, number, number] {
  const v = evalVector3(vector);
  return [
    naiveEval(v.x.n, valuedVars),
    naiveEval(v.y.n, valuedVars),
    naiveEval(v.z.n, valuedVars),
  ];
}

export function findSolution(
  constraints: ConstraintNode[],
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
  let loss = evalConstraint(constraints[0]);

  constraints.slice(1).forEach((term) => {
    loss = loss.add(evalConstraint(term));
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

export function exportAsFidget(node: SolidNode | ProfileNode) {
  const generic2dPoint = vecFromCartesianCoords(NumX, NumY).pointFromOrigin();
  const generic3dPoint = vec3FromCartesianCoords(
    NumX,
    NumY,
    NumZ,
  ).pointFromOrigin();

  console.log("exporting");

  const distNode =
    node instanceof SolidNode
      ? evalSolid(node).valueAt(generic3dPoint)
      : evalProfile(node).distanceTo(generic2dPoint);
  return fidgetStringify(distNode.n);
}
