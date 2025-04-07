import {
  DistanceLiteral,
  PointVectorSum,
  VectorFromCartesianCoords,
  VectorFromPolarCoods,
  VectorDifference,
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
  readRealValue,
  VectorRotated,
  AnyRealValueNode,
  AnyDistanceNode,
  AnyAngleNode,
  AnyVectorNode,
  AnyPointNode,
} from "sketch";
import {
  AngleLike,
  asAngle,
  asDistance,
  asPoint,
  asRealValue,
  asVector,
  DistanceLike,
  PointLike,
  RealLike,
  VectorLike,
} from "./convert";

import { NodeWrapper } from "./types";

export class Real implements NodeWrapper<AnyRealValueNode> {
  constructor(public inner: AnyRealValueNode) {}

  read(variables: Map<string, number>): number {
    return readRealValue(this.inner, variables);
  }
}

export class Distance implements NodeWrapper<AnyDistanceNode> {
  constructor(public inner: AnyDistanceNode) {}

  read(variables: Map<string, number>): number {
    return readDistance(this.inner, variables);
  }
}

export class Angle implements NodeWrapper<AnyAngleNode> {
  constructor(public inner: AnyAngleNode) {}

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
}

export class Vector implements NodeWrapper<AnyVectorNode> {
  constructor(readonly inner: AnyVectorNode) {}

  public add(other: VectorLike): Vector {
    return new Vector(new VectorSum(this.inner, asVector(other)));
  }

  public subtract(other: VectorLike): Vector {
    return new Vector(new VectorDifference(this.inner, asVector(other)));
  }

  public scale(factor: RealLike): Vector {
    return new Vector(new VectorScaled(this.inner, asRealValue(factor)));
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

  public rotate(angle: AngleLike): Vector {
    return new Vector(new VectorRotated(this.inner, asAngle(angle)));
  }
}

export class Point implements NodeWrapper<AnyPointNode> {
  constructor(public inner: AnyPointNode) {}

  public translate(vector: VectorLike): Point {
    return new Point(new PointVectorSum(this.inner, asVector(vector)));
  }

  public translateX(x: RealLike): Point {
    return new Point(
      new PointVectorSum(
        this.inner,
        new VectorFromCartesianCoords(asRealValue(x), 0),
      ),
    );
  }

  public translateY(y: RealLike): Point {
    return new Point(
      new PointVectorSum(
        this.inner,
        new VectorFromCartesianCoords(0, asRealValue(y)),
      ),
    );
  }

  translatePolar(angle: AngleLike, radius: DistanceLike = 1): Point {
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

  public trX(x: RealLike): Point {
    return this.translateX(x);
  }

  public trY(y: RealLike): Point {
    return this.translateY(y);
  }

  public trPolar(angle: AngleLike, radius: DistanceLike = 1): Point {
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

  public rotateAround(angle: AngleLike, center: PointLike): Point {
    const c = point(center);
    const v = this.vecFrom(c);
    return c.translate(v.rotate(angle));
  }
}

export function angle(degrees: AngleLike): Angle {
  if (degrees instanceof Angle) {
    return degrees;
  }
  return new Angle(asAngle(degrees));
}

export function distance(value: DistanceLike): Distance {
  if (value instanceof Distance) {
    return value;
  }
  return new Distance(asDistance(value));
}

export function vector(value: VectorLike): Vector {
  if (value instanceof Vector) {
    return value;
  }
  return new Vector(asVector(value));
}

export function point(value: PointLike): Point {
  if (value instanceof Point) {
    return value;
  }
  return new Point(asPoint(value));
}
