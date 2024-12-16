import {
  AngleLiteral,
  AngleNode,
  DistanceLiteral,
  DistanceNode,
  PointAsVectorFromOrigin,
  PointNode,
  ProfileNode,
  RealValueNode,
  VectorFromCartesianCoords,
  VectorFromPolarCoods,
  VectorNode,
} from "sketch";
import { isNodeWrapper, NodeWrapper } from "./types";

export type RealLike = number | RealValueNode | NodeWrapper<RealValueNode>;

export function asRealValue(value: RealLike): RealValueNode {
  if (value instanceof RealValueNode) {
    return value;
  }
  if (isNodeWrapper(value, RealValueNode)) {
    return value.inner;
  }
  return value;
}

export type DistanceLike = number | DistanceNode | NodeWrapper<DistanceNode>;

export function asDistance(distance: DistanceLike): DistanceNode {
  if (distance instanceof DistanceNode) {
    return distance;
  }
  if (isNodeWrapper(distance, DistanceNode)) {
    return distance.inner;
  }

  return new DistanceLiteral(distance);
}

export function asDistanceOrUndefined(
  value: DistanceLike | undefined,
): DistanceNode | undefined {
  return value || value === 0 ? asDistance(value) : value;
}

export type AngleLike = number | AngleNode | NodeWrapper<AngleNode>;

export function asAngle(angle: AngleLike): AngleNode {
  if (angle instanceof AngleNode) {
    return angle;
  }
  if (isNodeWrapper(angle, AngleNode)) {
    return angle.inner;
  }
  return new AngleLiteral(angle);
}

export type VectorLike =
  | [RealLike, RealLike]
  | VectorNode
  | NodeWrapper<VectorNode>;

export function asVector(vector: VectorLike): VectorNode {
  if (vector instanceof VectorNode) {
    return vector;
  }
  if (isNodeWrapper(vector, VectorNode)) {
    return vector.inner;
  }

  if (isNodeWrapper(vector, PointNode) || vector instanceof PointNode) {
    throw new Error("Expected a Vector, but recieved a Point");
  }

  const [x, y] = vector;
  return new VectorFromCartesianCoords(asRealValue(x), asRealValue(y));
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

export type PointLike =
  | [RealLike, RealLike]
  | PointNode
  | NodeWrapper<PointNode>;

export function asPoint(point: PointLike): PointNode {
  if (point instanceof PointNode) {
    return point;
  }
  if (isNodeWrapper(point, PointNode)) {
    return point.inner;
  }

  if (isNodeWrapper(point, VectorNode) || point instanceof VectorNode) {
    throw new Error("Expected a Point, but recieved a Vector");
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

export type ProfileLike = ProfileNode | NodeWrapper<ProfileNode>;

export function asProfile(profile: ProfileLike): ProfileNode {
  if (profile instanceof ProfileNode) {
    return profile;
  }
  if (isNodeWrapper(profile, ProfileNode)) {
    return profile.inner;
  }
  throw new Error("Expected a ProfileNode");
}
