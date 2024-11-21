import {
  AngleLiteral,
  AngleNode,
  DistanceLiteral,
  DistanceNode,
  PointAsVectorFromOrigin,
  PointNode,
  VectorFromCartesianCoords,
  VectorFromPolarCoods,
  VectorNode,
} from "sketch";
import { isNodeWrapper, NodeWrapper } from "./types";

export function asDistance(distance: number | DistanceNode): DistanceNode {
  if (distance instanceof DistanceNode) {
    return distance;
  }
  return new DistanceLiteral(distance);
}

export function asAngle(angle: number | AngleNode): AngleNode {
  if (angle instanceof AngleNode) {
    return angle;
  }
  return new AngleLiteral(angle);
}

export type VectorLike =
  | [number, number]
  | VectorNode
  | NodeWrapper<VectorNode>;

export function asVector(vector: VectorLike): VectorNode {
  if (vector instanceof VectorNode) {
    return vector;
  }
  if (isNodeWrapper(vector, VectorNode)) {
    return vector.inner;
  }
  const [x, y] = vector;
  return new VectorFromCartesianCoords(x, y);
}

export function asPolarVector(vector: VectorLike): VectorNode {
  if (vector instanceof VectorNode) {
    return vector;
  }
  if (isNodeWrapper(vector, VectorNode)) {
    return vector.inner;
  }
  const [angle, radius] = vector;
  return new VectorFromPolarCoods(asDistance(radius), asAngle(angle));
}

export type PointLike = [number, number] | PointNode | NodeWrapper<PointNode>;

export function asPoint(point: PointLike): PointNode {
  console.log("asPoint", point, isNodeWrapper(point, PointNode));
  if (point instanceof PointNode) {
    return point;
  }
  if (isNodeWrapper(point, PointNode)) {
    return point.inner;
  }
  return new PointAsVectorFromOrigin(asVector(point));
}

export function asPolarPoint(point: PointLike): PointNode {
  if (point instanceof PointNode) {
    return point;
  }
  if (isNodeWrapper(point, PointNode)) {
    return point.inner;
  }
  return new PointAsVectorFromOrigin(asPolarVector(point));
}
