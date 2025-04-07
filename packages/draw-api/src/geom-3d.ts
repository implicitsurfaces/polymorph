import {
  PivotedPlaneNode,
  AnyPlaneNode,
  Point3AsVectorFromOrigin,
  Point3MidPoint,
  AnyPoint3Node,
  Point3VectorSum,
  readPoint3,
  readVector3,
  RotatedPlaneNode,
  TranslatedPlaneNode,
  Vector3Difference,
  Vector3FromPoint,
  Vector3FromPoints,
  AnyVector3Node,
  Vector3Norm,
  Vector3Rotated,
  Vector3Scaled,
  Vector3Sum,
} from "sketch";
import { NodeWrapper } from "./types";
import {
  AngleLike,
  asAngle,
  asDistance,
  asPlane,
  asPoint3D,
  asVector3D,
  DistanceLike,
  PlaneLike,
  Point3DLike,
  RealLike,
  Vector3DLike,
} from "./convert";
import { Distance } from "./geom";

export class Point3D implements NodeWrapper<AnyPoint3Node> {
  constructor(public inner: AnyPoint3Node) {}

  public translate(vector: Vector3DLike): Point3D {
    return new Point3D(new Point3VectorSum(this.inner, asVector3D(vector)));
  }

  public translateX(x: RealLike): Point3D {
    return this.translate([x, 0, 0]);
  }

  public translateY(y: RealLike): Point3D {
    return this.translate([0, y, 0]);
  }

  public translateZ(z: RealLike): Point3D {
    return this.translate([0, 0, z]);
  }

  public tr(vect: Vector3DLike): Point3D {
    return this.translate(vect);
  }

  public trX(x: RealLike): Point3D {
    return this.translateX(x);
  }

  public trY(y: RealLike): Point3D {
    return this.translateY(y);
  }

  public trZ(z: RealLike): Point3D {
    return this.translateZ(z);
  }

  public vecTo(other: Point3DLike): Vector3D {
    return new Vector3D(new Vector3FromPoints(this.inner, asPoint3D(other)));
  }

  public vecFrom(other: Point3DLike): Vector3D {
    return new Vector3D(new Vector3FromPoints(asPoint3D(other), this.inner));
  }

  public vecFromOrigin(): Vector3D {
    return new Vector3D(new Vector3FromPoint(this.inner));
  }

  public midPoint(other: Point3DLike): Point3D {
    return new Point3D(new Point3MidPoint(this.inner, asPoint3D(other)));
  }

  public read(variables: Map<string, number>): [number, number, number] {
    return readPoint3(this.inner, variables);
  }
}

export function point3D(p: Point3DLike): Point3D {
  if (p instanceof Point3D) {
    return p;
  }
  return new Point3D(asPoint3D(p));
}

export class Vector3D implements NodeWrapper<AnyVector3Node> {
  constructor(public inner: AnyVector3Node) {}

  public add(other: Vector3DLike): Vector3D {
    return new Vector3D(new Vector3Sum(this.inner, asVector3D(other)));
  }

  public subtract(other: Vector3DLike): Vector3D {
    return new Vector3D(new Vector3Difference(this.inner, asVector3D(other)));
  }

  public scale(factor: DistanceLike): Vector3D {
    return new Vector3D(new Vector3Scaled(this.inner, asDistance(factor)));
  }

  public norm(): Distance {
    return new Distance(new Vector3Norm(this.inner));
  }

  public toPoint(): Point3D {
    return new Point3D(new Point3AsVectorFromOrigin(this.inner));
  }

  public rotate(
    angle: AngleLike,
    axis: "x" | "y" | "z" | Vector3DLike = "x",
  ): Vector3D {
    const a =
      axis === "x" || axis === "y" || axis === "z" ? axis : asVector3D(axis);
    return new Vector3D(new Vector3Rotated(this.inner, asAngle(angle), a));
  }

  public read(variables: Map<string, number>): [number, number, number] {
    return readVector3(this.inner, variables);
  }
}

export function vector3D(v: Vector3DLike): Vector3D {
  if (v instanceof Vector3D) {
    return v;
  }
  return new Vector3D(asVector3D(v));
}

export class Plane implements NodeWrapper<AnyPlaneNode> {
  constructor(public inner: AnyPlaneNode) {}

  public translate(vector: Vector3DLike): Plane {
    return new Plane(new TranslatedPlaneNode(this.inner, asVector3D(vector)));
  }

  public translateX(x: RealLike): Plane {
    return this.translate([x, 0, 0]);
  }

  public translateY(y: RealLike): Plane {
    return this.translate([0, y, 0]);
  }

  public translateZ(z: RealLike): Plane {
    return this.translate([0, 0, z]);
  }

  public tr(vect: Vector3DLike): Plane {
    return this.translate(vect);
  }

  public trX(x: RealLike): Plane {
    return this.translateX(x);
  }

  public trY(y: RealLike): Plane {
    return this.translateY(y);
  }

  public trZ(z: RealLike): Plane {
    return this.translateZ(z);
  }

  public pivot(
    angle: AngleLike,
    axis: "x" | "y" | "z" | Vector3DLike = "x",
  ): Plane {
    const a =
      axis === "x" || axis === "y" || axis === "z" ? axis : asVector3D(axis);
    return new Plane(new PivotedPlaneNode(this.inner, asAngle(angle), a));
  }

  public rotate(angle: AngleLike): Plane {
    return new Plane(new RotatedPlaneNode(this.inner, asAngle(angle)));
  }
}

export function plane(p: PlaneLike = "xy"): Plane {
  if (p instanceof Plane) {
    return p;
  }
  return new Plane(asPlane(p));
}
