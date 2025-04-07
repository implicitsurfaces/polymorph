import type { Point } from "../geom";
import { asNum, Num, variable } from "../num";
import { AnyRealValueNode } from "../sketch-nodes";
import { DistField } from "../types";

export function isRealValueNode(value: unknown): value is AnyRealValueNode {
  return (
    typeof value === "object" &&
    value !== null &&
    "nodeType" in value &&
    value.nodeType === "RealValueVariable"
  );
}

export function realValueChildren(value: AnyRealValueNode | number) {
  if (typeof value === "number") {
    return [];
  }

  if (value.nodeType === "RealValueVariable") {
    return [];
  }

  if (value.nodeType === "SignedDistanceToProfile") {
    return [value.profile, value.point];
  }
}

export function evalRealValue(
  value: AnyRealValueNode | number,
  children: unknown[],
): Num {
  if (typeof value === "number") {
    return asNum(value);
  }

  if (value.nodeType === "RealValueVariable") {
    return variable(value.name);
  }

  if (typeof value === "number") {
    return asNum(value);
  }

  if (value.nodeType === "SignedDistanceToProfile") {
    return (children[0] as DistField).distanceTo(children[1] as Point);
  }

  throw new Error(`Unknown real value: ${value}`);
}

export const realValueHandler = {
  isNode: isRealValueNode,
  children: realValueChildren,
  eval: evalRealValue,
};
