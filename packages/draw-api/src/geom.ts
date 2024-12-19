import {
  AngleNode,
  DistanceLiteral,
  PointNode,
  PointVectorSum,
  VectorFromCartesianCoords,
  VectorFromPolarCoods,
  DistanceNode,
  VectorDifference,
  VectorNode,
  VectorScaled,
  VectorSum,
  VectorFromPoint,
  VectorFromPoints,
  PointMidPoint,
  AngleSum,
  AngleDifference,
  AngleBisection,
  VectorDirection,
  AnglePerpendicular,
  AngleOpposite,
  VectorNorm,
  readDistance,
  readAngleAsDegree,
  readVector,
  readPoint,
  PointAsVectorFromOrigin,
  RealValueNode,
  readRealValue,
  VectorRotated,
  treeReprRealValue,
  treeReprDistance,
  treeReprAngle,
  treeReprVector,
  treeReprPoint,
} from "sketch";
import {
  asAngle,
  asDistance,
  asPoint,
  asVector,
  PointLike,
  VectorLike,
} from "./convert";

import { NodeWrapper } from "./types";

export class Real implements NodeWrapper<RealValueNode> {
  constructor(public inner: RealValueNode) {}

  read(variables: Map<string, number>): number {
    return readRealValue(this.inner, variables);
  }

  treeRepr(): string {
    return treeReprRealValue(this.inner);
  }
}

export class Distance implements NodeWrapper<DistanceNode> {
  constructor(public inner: DistanceNode) {}

  read(variables: Map<string, number>): number {
    return readDistance(this.inner, variables);
  }

  treeRepr(): string {
    return treeReprDistance(this.inner);
  }
}

export class Angle implements NodeWrapper<AngleNode> {
  constructor(public inner: AngleNode) {}

  public add(other: Angle): Angle {
    return new Angle(new AngleSum(this.inner, other.inner));
  }

  public subtract(other: Angle): Angle {
    return new Angle(new AngleDifference(this.inner, other.inner));
  }

  public bisect(): Angle {
    return new Angle(new AngleBisection(this.inner));
  }

  public perp(): Angle {
    return new Angle(new AnglePerpendicular(this.inner));
  }

  public opposite(): Angle {
    return new Angle(new AngleOpposite(this.inner));
  }

  public asVec(): Vector {
    return new Vector(
      new VectorFromPolarCoods(new DistanceLiteral(1), this.inner),
    );
  }

  public read(variables: Map<string, number>): number {
    return readAngleAsDegree(this.inner, variables);
  }

  public treeRepr(): string {
    return treeReprAngle(this.inner);
  }
}

export class Vector implements NodeWrapper<VectorNode> {
  constructor(readonly inner: VectorNode) {}

  public add(other: Vector): Vector {
    return new Vector(new VectorSum(this.inner, other.inner));
  }

  public subtract(other: Vector): Vector {
    return new Vector(new VectorDifference(this.inner, other.inner));
  }

  public scale(factor: number | DistanceNode): Vector {
    return new Vector(new VectorScaled(this.inner, asDistance(factor)));
  }

  public asAngle(): Angle {
    return new Angle(new VectorDirection(this.inner));
  }

  public norm(): Distance {
    return new Distance(new VectorNorm(this.inner));
  }

  public read(variables: Map<string, number>): [number, number] {
    return readVector(this.inner, variables);
  }

  public toPoint(): Point {
    return new Point(new PointAsVectorFromOrigin(this.inner));
  }

  public rotate(angle: number | AngleNode): Vector {
    return new Vector(new VectorRotated(this.inner, asAngle(angle)));
  }

  public treeRepr(): string {
    return treeReprVector(this.inner);
  }
}

export class Point implements NodeWrapper<PointNode> {
  constructor(public inner: PointNode) {}

  public translate(vector: VectorLike): Point {
    return new Point(new PointVectorSum(this.inner, asVector(vector)));
  }

  public translateX(x: number | DistanceNode): Point {
    return new Point(
      new PointVectorSum(
        this.inner,
        new VectorFromCartesianCoords(asDistance(x), new DistanceLiteral(0)),
      ),
    );
  }

  public translateY(y: number | DistanceNode): Point {
    return new Point(
      new PointVectorSum(
        this.inner,
        new VectorFromCartesianCoords(new DistanceLiteral(0), asDistance(y)),
      ),
    );
  }

  translatePolar(
    angle: number | AngleNode,
    radius: number | DistanceNode = 1,
  ): Point {
    return new Point(
      new PointVectorSum(
        this.inner,
        new VectorFromPolarCoods(asDistance(radius), asAngle(angle)),
      ),
    );
  }

  public tr(vect: VectorLike): Point {
    return this.translate(vect);
  }

  public trX(x: number | DistanceNode): Point {
    return this.translateX(x);
  }

  public trY(y: number | DistanceNode): Point {
    return this.translateY(y);
  }

  public trPolar(
    angle: number | AngleNode,
    radius: number | DistanceNode = 1,
  ): Point {
    return this.translatePolar(angle, radius);
  }

  public vecTo(other: Point): Vector {
    return new Vector(new VectorFromPoints(this.inner, other.inner));
  }

  public vecFrom(other: Point): Vector {
    return new Vector(new VectorFromPoints(other.inner, this.inner));
  }

  public vecFromOrigin(): Vector {
    return new Vector(new VectorFromPoint(this.inner));
  }

  public midPoint(other: Point): Point {
    return new Point(new PointMidPoint(this.inner, other.inner));
  }

  public read(variables: Map<string, number>): [number, number] {
    return readPoint(this.inner, variables);
  }

  public rotateAround(angle: number | AngleNode, center: PointLike): Point {
    const c = point(center);
    const v = this.vecFrom(c);
    return c.translate(v.rotate(angle));
  }

  public treeRepr(): string {
    return treeReprPoint(this.inner);
  }
}

export function angle(degrees: number | Angle | AngleNode): Angle {
  if (degrees instanceof Angle) {
    return degrees;
  }
  return new Angle(asAngle(degrees));
}

export function distance(value: number | Distance | DistanceNode): Distance {
  if (value instanceof Distance) {
    return value;
  }
  return new Distance(asDistance(value));
}

export function vector(value: VectorLike | Vector): Vector {
  if (value instanceof Vector) {
    return value;
  }
  return new Vector(asVector(value));
}

export function point(value: PointLike | Point): Point {
  if (value instanceof Point) {
    return value;
  }
  return new Point(asPoint(value));
}
