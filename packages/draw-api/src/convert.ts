import {
  AngleLiteral,
  AnyAngleNode,
  AnyDistanceNode,
  AnyPlaneNode,
  AnyPoint3Node,
  AnyPointNode,
  AnyRealValueNode,
  AnyVector3Node,
  AnyVectorNode,
  BasePlaneNode,
  DistanceLiteral,
  DistanceNode,
  PlaneFromPoints,
  Point3AsVectorFromOrigin,
  PointAsVectorFromOrigin,
  Vector3FromCartesianCoords,
  VectorFromCartesianCoords,
  VectorFromPolarCoods,
  AnyProfileNode,
} from "sketch";
import { isNodeWrapper, isOfCategory, NodeWrapper } from "./types";

export type DistanceLike =
  | number
  | AnyDistanceNode
  | NodeWrapper<AnyDistanceNode>;

export function asDistance(distance: DistanceLike): AnyDistanceNode {
  if (isOfCategory(distance, "Distance")) {
    return distance;
  }
  if (isNodeWrapper(distance, "Distance")) {
    return distance.inner;
  }

  return new DistanceLiteral(distance);
}

export type RealLike =
  | number
  | AnyRealValueNode
  | NodeWrapper<AnyRealValueNode>
  | AnyDistanceNode
  | NodeWrapper<AnyDistanceNode>;

export function asRealValue(value: RealLike): AnyRealValueNode {
  if (isOfCategory(value, "Distance")) {
    return value;
  }

  if (isNodeWrapper(value, "Distance")) {
    return value.inner;
  }

  if (isOfCategory(value, "RealValue")) {
    return value;
  }

  if (isNodeWrapper(value, "RealValue")) {
    return value.inner;
  }

  if (typeof value === "number") {
    return value;
  }

  throw new Error("Expected a RealValue or Distance, but received: " + value);
}

export function asDistanceOrUndefined(
  value: DistanceLike | undefined,
): DistanceNode | undefined {
  return value === undefined ? value : asDistance(value);
}

export type AngleLike = number | AnyAngleNode | NodeWrapper<AnyAngleNode>;

export function asAngle(angle: AngleLike): AnyAngleNode {
  if (typeof angle === "number") {
    return new AngleLiteral(angle);
  }
  if (isNodeWrapper(angle, "Angle")) {
    return angle.inner;
  }

  return angle;
}

export type VectorLike =
  | [RealLike, RealLike]
  | AnyVectorNode
  | NodeWrapper<AnyVectorNode>;

export function asVector(vector: VectorLike): AnyVectorNode {
  if (isOfCategory(vector, "Vector")) {
    return vector;
  }
  if (isNodeWrapper(vector, "Vector")) {
    return vector.inner;
  }

  if (isNodeWrapper(vector, "Point") || isOfCategory(vector, "Point")) {
    throw new Error("Expected a Vector, but recieved a Point");
  }

  const [x, y] = vector;
  return new VectorFromCartesianCoords(asRealValue(x), asRealValue(y));
}

export type Vector3DLike =
  | [RealLike, RealLike, RealLike]
  | AnyVector3Node
  | NodeWrapper<AnyVector3Node>;

export function asVector3D(vector: Vector3DLike): AnyVector3Node {
  if (isOfCategory(vector, "Vector3")) {
    return vector;
  }
  if (isNodeWrapper(vector, "Vector3")) {
    return vector.inner;
  }

  if (isNodeWrapper(vector, "Point3") || isOfCategory(vector, "Point3")) {
    throw new Error("Expected a Vector, but recieved a Point");
  }

  const [x, y, z] = vector;
  return new Vector3FromCartesianCoords(
    asRealValue(x),
    asRealValue(y),
    asRealValue(z),
  );
}

export type PolarVectorLike =
  | [AngleLike, DistanceLike]
  | AnyVectorNode
  | NodeWrapper<AnyVectorNode>;
export function asPolarVector(vector: PolarVectorLike): AnyVectorNode {
  if (isOfCategory(vector, "Vector")) {
    return vector;
  }
  if (isNodeWrapper(vector, "Vector")) {
    return vector.inner;
  }
  const [angle, radius] = vector;
  return new VectorFromPolarCoods(asDistance(radius), asAngle(angle));
}

export type PointLike =
  | [RealLike, RealLike]
  | AnyPointNode
  | NodeWrapper<AnyPointNode>;

export function asPoint(point: PointLike): AnyPointNode {
  if (isOfCategory(point, "Point")) {
    return point;
  }
  if (isNodeWrapper(point, "Point")) {
    return point.inner;
  }

  if (isNodeWrapper(point, "Vector") || isOfCategory(point, "Vector")) {
    throw new Error("Expected a Point, but recieved a Vector");
  }

  return new PointAsVectorFromOrigin(asVector(point));
}

export type PolarPointLike =
  | [AngleLike, DistanceLike]
  | AnyPointNode
  | NodeWrapper<AnyPointNode>;
export function asPolarPoint(point: PolarPointLike): AnyPointNode {
  if (isOfCategory(point, "Point")) {
    return point;
  }
  if (isNodeWrapper(point, "Point")) {
    return point.inner;
  }
  return new PointAsVectorFromOrigin(asPolarVector(point));
}

export type Point3DLike =
  | [RealLike, RealLike, RealLike]
  | AnyPoint3Node
  | NodeWrapper<AnyPoint3Node>;

export function asPoint3D(point: Point3DLike): AnyPoint3Node {
  if (isOfCategory(point, "Point3")) {
    return point;
  }
  if (isNodeWrapper(point, "Point3")) {
    return point.inner;
  }
  return new Point3AsVectorFromOrigin(asVector3D(point));
}

export type ProfileLike = AnyProfileNode | NodeWrapper<AnyProfileNode>;

export function asProfile(profile: ProfileLike): AnyProfileNode {
  if (isOfCategory(profile, "Profile")) {
    return profile;
  }
  if (isNodeWrapper(profile, "Profile")) {
    return profile.inner;
  }
  throw new Error("Expected a ProfileNode");
}

export type PlaneLike =
  | AnyPlaneNode
  | NodeWrapper<AnyPlaneNode>
  | [Point3DLike, Point3DLike, Point3DLike]
  | "xy"
  | "yz"
  | "xz";

export function asPlane(plane: PlaneLike): AnyPlaneNode {
  if (isOfCategory(plane, "Plane")) {
    return plane;
  }
  if (isNodeWrapper(plane, "Plane")) {
    return plane.inner;
  }

  if (plane === "xy" || plane === "yz" || plane === "xz") {
    return new BasePlaneNode(plane);
  }

  const [origin, p1, p2] = plane;
  return new PlaneFromPoints(asPoint3D(origin), asPoint3D(p1), asPoint3D(p2));
}
