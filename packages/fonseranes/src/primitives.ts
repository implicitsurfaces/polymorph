import { angleBetweenVecs } from "./angle";
import {
  ColVec3,
  ColVec4,
  diagonalMatrix3x3,
  Matrix3x3,
  Matrix4x4,
} from "./matrices";
import {
  canonicalCoordinateSystem,
  PlanarCoordinateSystem,
} from "./plane-helpers";
import {
  Transform2D,
  Transform3D,
  Transformable2D,
  Transformable3D,
} from "./transforms";
import {
  asInPlaneTransformation,
  projectToPlaneInDirection,
  projectToPlaneWithPoint,
} from "./transforms-helpers";

export class Point2D extends Transformable2D<Point2D> {
  constructor(
    public readonly x: number,
    public readonly y: number,
  ) {
    super();
  }

  public static fromCoords(coords: { x: number; y: number }): Point2D {
    return new Point2D(coords.x, coords.y);
  }

  vecTo(other: Point2D): Vec2D {
    return new Vec2D(other.x - this.x, other.y - this.y);
  }

  vecFrom(other: Point2D): Vec2D {
    return new Vec2D(this.x - other.x, this.y - other.y);
  }

  sub(other: Vec2D): Point2D {
    return new Point2D(this.x - other.x, this.y - other.y);
  }

  add(other: Vec2D): Point2D {
    return new Point2D(this.x + other.x, this.y + other.y);
  }

  vecFromOrigin(): Vec2D {
    return new Vec2D(this.x, this.y);
  }

  midPoint(other: Point2D): Point2D {
    return new Point2D((this.x + other.x) / 2, (this.y + other.y) / 2);
  }

  asTuple(): [number, number] {
    return [this.x, this.y];
  }

  levelSet(p: Point2D): number {
    return this.vecTo(p).magnitude();
  }

  transform(transform: Transform2D): Point2D {
    const colVec = new ColVec3(this.x, this.y, 1);
    const result = transform.matrix.product(colVec);

    if (result.x3 === 1) {
      return new Point2D(result.x1, result.x2);
    }
    if (result.x3 === 0) {
      throw new Error("Invalid transformation: result is a point at infinity");
    }
    return new Point2D(result.x1 / result.x3, result.x2 / result.x3);
  }
}

export class Vec2D extends Transformable2D<Vec2D> {
  constructor(
    public readonly x: number,
    public readonly y: number,
  ) {
    super();
  }

  public static fromCoords(coords: { x: number; y: number }): Vec2D {
    return new Vec2D(coords.x, coords.y);
  }

  perp(): Vec2D {
    return new Vec2D(-this.y, this.x);
  }

  scale(scalar: number): Vec2D {
    return new Vec2D(this.x * scalar, this.y * scalar);
  }

  magnitude(): number {
    return Math.sqrt(this.x ** 2 + this.y ** 2);
  }

  asTuple(): [number, number] {
    return [this.x, this.y];
  }

  normalize(): Vec2D {
    const mag = this.magnitude();
    if (mag === 0) {
      throw new Error("Cannot normalize a zero vector");
    }
    return new Vec2D(this.x / mag, this.y / mag);
  }

  transform(transform: Transform2D): Vec2D {
    const colVec = new ColVec3(this.x, this.y, 0);
    const result = transform.matrix.product(colVec);
    if (result.x3 !== 0) {
      throw new Error("Invalid transformation: result is a point");
    }
    return new Vec2D(result.x1, result.x2);
  }
}

export class Line2D extends Transformable2D<Line2D> {
  constructor(
    public readonly a: number,
    public readonly b: number,
    public readonly c: number,
  ) {
    super();
  }

  static fromPoints(p1: Point2D, p2: Point2D): Line2D {
    const a = p2.y - p1.y;
    const b = p1.x - p2.x;
    const c = a * p1.x + b * p1.y;
    return new Line2D(a, b, -c);
  }

  static fromPointAndVec(p: Point2D, v: Vec2D): Line2D {
    const a = v.y;
    const b = -v.x;
    const c = a * p.x + b * p.y;
    return new Line2D(a, b, -c);
  }

  get normal(): Vec2D {
    return new Vec2D(this.a, this.b);
  }

  get direction(): Vec2D {
    return new Vec2D(-this.b, this.a);
  }

  segment(p1: Point2D, p2: Point2D): Segment2D {
    const normal = this.normal;

    const startLimit = Line2D.fromPointAndVec(p1, normal);
    const endLimit = Line2D.fromPointAndVec(p2, normal);

    return new Segment2D(this, [
      new HalfPlane2D(startLimit, false),
      new HalfPlane2D(endLimit, true),
    ]);
  }

  clip(line: Line2D, reverse = false): Segment2D {
    const clipSection = new HalfPlane2D(line, reverse);
    return new Segment2D(this, [clipSection]);
  }

  partition(line: Line2D, reverse = false): Partition2D {
    const clipSection = new HalfPlane2D(line, reverse);
    return new Partition2D(this, [clipSection]);
  }

  levelSet(p: Point2D): number {
    return this.a * p.x + this.b * p.y + this.c;
  }

  windingNumber(p: Point2D): number {
    const levelSet = this.levelSet(p);
    return Math.sign(levelSet) * 0.5;
  }

  transform(transform: Transform2D): Line2D {
    const matrix = transform.matrix.inverse().transpose();
    const colVec = new ColVec3(this.a, this.b, this.c);
    const result = matrix.product(colVec);
    return new Line2D(result.x1, result.x2, result.x3);
  }
}

export class HalfPlane2D extends Transformable2D<HalfPlane2D> {
  constructor(
    public readonly line: Line2D,
    public readonly inverse: boolean,
  ) {
    super();
  }

  isInside(p: Point2D): boolean {
    return this.levelSet(p) < 0;
  }

  levelSet(p: Point2D): number {
    const levelSet = this.line.levelSet(p);
    return this.inverse ? -levelSet : levelSet;
  }

  windingNumber(p: Point2D): number {
    const levelSet = this.levelSet(p);
    return Math.sign(levelSet) * 0.5;
  }

  transform(transform: Transform2D): HalfPlane2D {
    const line = this.line.transform(transform);
    return new HalfPlane2D(line, this.inverse);
  }
}

const Q_MAT = diagonalMatrix3x3(1, 1, -1);

export class Conic2D extends Transformable2D<Conic2D> {
  private _matrix: Matrix3x3 | null = null;
  constructor(public readonly transformation: Transform2D) {
    super();
  }

  segment(p1: Point2D, p2: Point2D): Segment2D {
    const line = Line2D.fromPoints(p2, p1);
    return this.clip(line);
  }

  clip(line: Line2D, reverse = false): Segment2D {
    const clipSection = new HalfPlane2D(line, reverse);
    return new Segment2D(this, [clipSection]);
  }

  partition(line: Line2D, reverse = false): Partition2D {
    const clipSection = new HalfPlane2D(line, reverse);
    return new Partition2D(this, [clipSection]);
  }

  get matrix(): Matrix3x3 {
    if (!this._matrix) {
      this._matrix = this.transformation.matrix
        .transpose()
        .mul(Q_MAT)
        .mul(this.transformation.matrix);
    }

    return this._matrix;
  }

  levelSet(p: Point2D): number {
    const colVec = new ColVec3(p.x, p.y, 1);
    return colVec.transpose().dot(this.matrix.product(colVec));
  }

  windingNumber(p: Point2D): number {
    const levelSet = this.levelSet(p);
    return Math.sign(levelSet);
  }

  transform(transform: Transform2D): Conic2D {
    const newTransform = this.transformation.compose(transform.reverse());
    return new Conic2D(newTransform);
  }

  contains(p: Point2D): boolean {
    const levelSet = this.levelSet(p);
    return levelSet < 0;
  }

  orientationSign = 1;
}

export type PrimitiveCurve2D = Line2D | Conic2D;

function insideHalfPlanes(point: Point2D, halfPlanes: HalfPlane2D[]): boolean {
  return halfPlanes.every((halfPlane) => halfPlane.isInside(point));
}

export class Partition2D extends Transformable2D<Partition2D> {
  constructor(
    public readonly curve: Curve2D,
    public readonly clipSections: HalfPlane2D[],
  ) {
    super();

    if (curve instanceof Partition2D) {
      this.curve = curve.curve;
      this.clipSections = [...curve.clipSections, ...clipSections];
    }
  }

  clip(line: Line2D, reverse = false): Segment2D {
    const clipSection = new HalfPlane2D(line, reverse);
    return new Segment2D(this, [clipSection]);
  }

  partition(line: Line2D, reverse = false): Partition2D {
    const clipSection = new HalfPlane2D(line, reverse);
    return new Partition2D(this.curve, [...this.clipSections, clipSection]);
  }

  transform(transform: Transform2D): Partition2D {
    const curve = this.curve.transform(transform);
    const section = this.clipSections.map((halfPlane) =>
      halfPlane.transform(transform),
    );
    return new Partition2D(curve, section);
  }

  levelSet(p: Point2D): number {
    return insideHalfPlanes(p, this.clipSections) ? this.curve.levelSet(p) : 10;
  }
}

export class Segment2D extends Transformable2D<Segment2D> {
  constructor(
    public readonly curve: Curve2D,
    public readonly clipSections: HalfPlane2D[],
  ) {
    super();
  }

  clip(line: Line2D, reverse = false): Segment2D {
    const clipSection = new HalfPlane2D(line, reverse);
    return new Segment2D(this.curve, [...this.clipSections, clipSection]);
  }

  transform(transform: Transform2D): Segment2D {
    const curve = this.curve.transform(transform);
    const section = this.clipSections.map((halfPlane) =>
      halfPlane.transform(transform),
    );
    return new Segment2D(curve, section);
  }

  levelSet(p: Point2D): number {
    return insideHalfPlanes(p, this.clipSections)
      ? Math.abs(this.curve.levelSet(p))
      : 10;
  }
}

export class ReversedConic2D extends Conic2D {
  constructor(public readonly transformation: Transform2D) {
    super(transformation);
  }

  clip(line: Line2D, reverse = false): Segment2D {
    const clipSection = new HalfPlane2D(line, !reverse);
    return new Segment2D(this, [clipSection]);
  }

  levelSet(p: Point2D): number {
    return -super.levelSet(p);
  }

  transform(transform: Transform2D): ReversedConic2D {
    const newTransform = this.transformation.compose(transform.reverse());
    return new ReversedConic2D(newTransform);
  }

  orientationSign = -1;
  contains(p: Point2D): boolean {
    const levelSet = this.levelSet(p);
    return levelSet > 0;
  }
}

export class ClosedPath2D extends Transformable2D<ClosedPath2D> {
  private _segments: Segment2D[] | null = null;
  private _orientationSign: number | null = null;
  constructor(
    public readonly points: Point2D[],
    public readonly curves: (PrimitiveCurve2D | ReversedConic2D)[],
  ) {
    super();
  }

  get orientationSign(): number {
    if (!this._orientationSign) {
      const surfaceArea = this.points.reduce((acc, point, i) => {
        const nextPoint = this.points[(i + 1) % this.points.length];
        return acc + (point.x * nextPoint.y - point.y * nextPoint.x);
      }, 0);

      this._orientationSign = Math.sign(surfaceArea);
    }

    return this._orientationSign;
  }

  get segments(): Segment2D[] {
    if (!this._segments) {
      this._segments = this.curves.map((curve, i) =>
        curve.segment(
          this.points[i],
          this.points[(i + 1) % this.points.length],
        ),
      );
    }
    return this._segments!;
  }

  windingNumber(p: Point2D): number {
    const windingNumbers = this.curves.map((curve, i) => {
      const p0 = this.points[i];
      const p1 = this.points[(i + 1) % this.points.length];

      const v1 = p.vecTo(p0);
      const v2 = p.vecTo(p1);

      if (v1.magnitude() === 0 || v2.magnitude() === 0) {
        return 0;
      }

      const chordWindingNumber =
        angleBetweenVecs(v1.x, v1.y, v2.x, v2.y).asRad() / (2 * Math.PI);

      if (curve instanceof Line2D) {
        return chordWindingNumber;
      }

      // We inverse the orientation sign if we are within the curve and the
      // chord and curve have different orientations

      const isInside = curve.contains(p);

      if (isInside && Math.sign(chordWindingNumber) !== curve.orientationSign) {
        return chordWindingNumber - this.orientationSign;
      } else {
        return chordWindingNumber;
      }
    });

    return windingNumbers.reduce((acc, val) => acc + val, 0);
  }

  levelSet(p: Point2D): number {
    const levels = this.segments.map((segment) => segment.levelSet(p));
    const chordLevelSet = Math.min(...levels);

    const windingNumber = this.windingNumber(p);

    const isInside = Math.abs(windingNumber) < 0.5;

    return isInside ? chordLevelSet : -chordLevelSet;
  }

  transform(transform: Transform2D): ClosedPath2D {
    const newSegments = this.curves.map((curve) => curve.transform(transform));
    const newPoints = this.points.map((point) => point.transform(transform));
    return new ClosedPath2D(newPoints, newSegments);
  }
}

export type Curve2D = Line2D | Conic2D | ClosedPath2D | Partition2D;

export class Point3D extends Transformable3D<Point3D> {
  constructor(
    public readonly x: number,
    public readonly y: number,
    public readonly z: number,
  ) {
    super();
  }

  public static fromCoords(coords: {
    x: number;
    y: number;
    z: number;
  }): Point3D {
    return new Point3D(coords.x, coords.y, coords.z);
  }

  vecTo(other: Point3D): Vec3D {
    return new Vec3D(other.x - this.x, other.y - this.y, other.z - this.z);
  }

  vecFrom(other: Point3D): Vec3D {
    return new Vec3D(this.x - other.x, this.y - other.y, this.z - other.z);
  }

  sub(other: Vec3D): Point3D {
    return new Point3D(this.x - other.x, this.y - other.y, this.z - other.z);
  }

  add(other: Vec3D): Point3D {
    return new Point3D(this.x + other.x, this.y + other.y, this.z + other.z);
  }

  vecFromOrigin(): Vec3D {
    return new Vec3D(this.x, this.y, this.z);
  }

  midPoint(other: Point3D): Point3D {
    return new Point3D(
      (this.x + other.x) / 2,
      (this.y + other.y) / 2,
      (this.z + other.z) / 2,
    );
  }

  transform(transform: Transform3D): Point3D {
    const colVec = new ColVec4(this.x, this.y, this.z, 1);
    const result = transform.matrix.product(colVec);

    if (result.x4 === 1) {
      return new Point3D(result.x1, result.x2, result.x3);
    }
    if (result.x4 === 0) {
      throw new Error("Invalid transformation: result is a point at infinity");
    }
    return new Point3D(
      result.x1 / result.x4,
      result.x2 / result.x4,
      result.x3 / result.x4,
    );
  }

  asTuple(): [number, number, number] {
    return [this.x, this.y, this.z];
  }
}

export class Vec3D extends Transformable3D<Vec3D> {
  constructor(
    public readonly x: number,
    public readonly y: number,
    public readonly z: number,
  ) {
    super();
  }

  public static fromCoords(coords: { x: number; y: number; z: number }): Vec3D {
    return new Vec3D(coords.x, coords.y, coords.z);
  }

  transform(transform: Transform3D): Vec3D {
    const colVec = new ColVec4(this.x, this.y, this.z, 0);
    const result = transform.matrix.product(colVec);

    if (result.x4 !== 0) {
      throw new Error("Invalid transformation: result is a point");
    }
    return new Vec3D(result.x1, result.x2, result.x3);
  }

  scale(scalar: number): Vec3D {
    return new Vec3D(this.x * scalar, this.y * scalar, this.z * scalar);
  }

  magnitude(): number {
    return Math.sqrt(this.magnitudeSquared());
  }

  magnitudeSquared(): number {
    return this.x ** 2 + this.y ** 2 + this.z ** 2;
  }

  normalize(): Vec3D {
    const mag = this.magnitude();
    if (mag === 0) {
      throw new Error("Cannot normalize a zero vector");
    }
    return new Vec3D(this.x / mag, this.y / mag, this.z / mag);
  }

  cross(other: Vec3D): Vec3D {
    return new Vec3D(
      this.y * other.z - this.z * other.y,
      this.z * other.x - this.x * other.z,
      this.x * other.y - this.y * other.x,
    );
  }

  asTuple(): [number, number, number] {
    return [this.x, this.y, this.z];
  }
}

export class Plane extends Transformable3D<Plane> {
  _coordinateSystem: PlanarCoordinateSystem | null = null;
  constructor(
    public readonly a: number,
    public readonly b: number,
    public readonly c: number,
    public readonly d: number,
  ) {
    super();
  }

  static fromPointAndNormal(point: Point3D, normal: Vec3D): Plane {
    return new Plane(
      normal.x,
      normal.y,
      normal.z,
      -(normal.x * point.x + normal.y * point.y + normal.z * point.z),
    );
  }

  get normal(): Vec3D {
    return new Vec3D(this.a, this.b, this.c);
  }

  asTuple(): [number, number, number, number] {
    return [this.a, this.b, this.c, this.d];
  }

  slice(slicePlane: Plane): Line2D {
    //const dir = this.normal.cross(slicePlane.normal);
    const dir = slicePlane.normal.cross(this.normal);
    if (dir.magnitudeSquared() === 0) {
      return new Line2D(0, 0, this.d - slicePlane.d);
    }

    const absX = Math.abs(dir.x);
    const absY = Math.abs(dir.y);
    const absZ = Math.abs(dir.z);

    let point: Point3D;
    if (absX >= absY && absX >= absZ) {
      // x is the largest component -> set x = 0
      const det = this.b * slicePlane.c - slicePlane.b * this.c;
      const y = (-this.d * slicePlane.c - -slicePlane.d * this.c) / det;
      const z = (this.b * -slicePlane.d - slicePlane.b * -this.d) / det;
      point = new Point3D(0, y, z);
    } else if (absY >= absX && absY >= absZ) {
      // y is largest -> set y = 0
      const det = this.a * slicePlane.c - slicePlane.a * this.c;
      const x = (-this.d * slicePlane.c - -slicePlane.d * this.c) / det;
      const z = (this.a * -slicePlane.d - slicePlane.a * -this.d) / det;
      point = new Point3D(x, 0, z);
    } else {
      // z is largest -> set z = 0
      const det = this.a * slicePlane.b - slicePlane.a * this.b;
      const x = (-this.d * slicePlane.b - -slicePlane.d * this.b) / det;
      const y = (this.a * -slicePlane.d - slicePlane.a * -this.d) / det;
      point = new Point3D(x, y, 0);
    }

    const projectedPoint = Point2D.fromCoords(
      slicePlane.coordinateSystem.globalToLocal(point),
    );
    const p2 = Point2D.fromCoords(
      slicePlane.coordinateSystem.globalToLocal(point.add(dir)),
    );

    return Line2D.fromPoints(projectedPoint, p2);
  }

  transform(transform: Transform3D): Plane {
    // For planes, we need to use the inverse transpose of the transformation matrix
    const invTranspose = transform.reverse().matrix.transpose();

    const planeRow = new ColVec4(this.a, this.b, this.c, this.d);

    const transformedRow = invTranspose.product(planeRow);

    return new Plane(
      transformedRow.x1,
      transformedRow.x2,
      transformedRow.x3,
      transformedRow.x4,
    );
  }

  translateNormal(distance: number): Plane {
    const normal = this.normal.normalize();
    return this.translate(normal.scale(distance));
  }

  levelSet(point: Point3D): number {
    return this.a * point.x + this.b * point.y + this.c * point.z + this.d;
  }

  pointInPlane(point: Point3D): boolean {
    const levelSet = this.levelSet(point);
    return Math.abs(levelSet) < 1e-10;
  }

  get coordinateSystem(): PlanarCoordinateSystem {
    if (!this._coordinateSystem) {
      this._coordinateSystem = canonicalCoordinateSystem(this.asTuple());
    }
    return this._coordinateSystem!;
  }

  embed(curve: Curve2D): Curve3D {
    return new Curve3D(curve, this);
  }
}

function planarTransfrom(
  fromPlane: Plane,
  toPlane: Plane,
  transformMatrix3D: Matrix4x4,
): Transform2D {
  const fromCoords = fromPlane.coordinateSystem.localToGlobalMatrix;
  const toCoords = toPlane.coordinateSystem.globalToLocalMatrix;

  const transformationMatrix = toCoords.mul(transformMatrix3D).mul(fromCoords);

  return new Transform2D(asInPlaneTransformation(transformationMatrix));
}

export function screenToScreenProjection(
  sourceScreen: Plane,
  targetScreen: Plane,
  direction: Vec3D,
): Transform2D {
  const projectionMatrix = projectToPlaneInDirection(
    targetScreen.asTuple(),
    direction.asTuple(),
  );

  return planarTransfrom(sourceScreen, targetScreen, projectionMatrix);
}

export function screenToScreenPerspectiveProjection(
  sourceScreen: Plane,
  targetScreen: Plane,
  apex: Point3D,
): Transform2D {
  const projectionMatrix = projectToPlaneWithPoint(
    targetScreen.asTuple(),
    apex.asTuple(),
  );

  return planarTransfrom(sourceScreen, targetScreen, projectionMatrix);
}

export class Curve3D extends Transformable3D<Curve3D> {
  constructor(
    public readonly curve: Curve2D,
    public readonly plane: Plane,
  ) {
    super();
  }

  projectIntoPlane(targetPlane: Plane): Curve2D {
    const transform = screenToScreenProjection(
      this.plane,
      targetPlane,
      targetPlane.normal,
    );
    return this.curve.transform(transform);
  }

  transform(transform: Transform3D): Curve3D {
    const newPlane = this.plane.transform(transform);

    const curveTransform = planarTransfrom(
      this.plane,
      newPlane,
      transform.matrix,
    );

    return new Curve3D(this.curve.transform(curveTransform), newPlane);
  }
}

export class Cylinder extends Transformable3D<Cylinder> {
  constructor(public readonly baseCurve: Curve3D) {
    super();
  }

  clip(plane: Plane, reverse = false): Patch {
    const clipSection = new HalfSpace3D(plane, reverse);
    return new Patch(this, [clipSection]);
  }

  partition(plane: Plane, reverse = false): Partition3D {
    const clipSection = new HalfSpace3D(plane, reverse);
    return new Partition3D(this, [clipSection]);
  }

  slice(slicePlane: Plane): Curve2D {
    const transform = screenToScreenProjection(
      this.baseCurve.plane,
      slicePlane,
      this.baseCurve.plane.normal,
    );
    return this.baseCurve.curve.transform(transform);
  }

  transform(transform: Transform3D): Cylinder {
    const newBaseCurve = this.baseCurve.transform(transform);
    return new Cylinder(newBaseCurve);
  }
}

export class Cone extends Transformable3D<Cone> {
  constructor(
    public readonly baseCurve: Curve3D,
    public readonly apex: Point3D,
  ) {
    super();
  }

  clip(plane: Plane, reverse = false): Patch {
    const clipSection = new HalfSpace3D(plane, reverse);
    return new Patch(this, [clipSection]);
  }

  partition(plane: Plane, reverse = false): Partition3D {
    const clipSection = new HalfSpace3D(plane, reverse);
    return new Partition3D(this, [clipSection]);
  }

  slice(slicePlane: Plane): Curve2D {
    const projectionMatrix = projectToPlaneWithPoint(slicePlane.asTuple(), [
      this.apex.x,
      this.apex.y,
      this.apex.z,
    ]);

    const transform = planarTransfrom(
      this.baseCurve.plane,
      slicePlane,
      projectionMatrix,
    );

    return this.baseCurve.curve.transform(transform);
  }

  transform(transform: Transform3D): Cone {
    const newBaseCurve = this.baseCurve.transform(transform);
    const newApex = this.apex.transform(transform);
    return new Cone(newBaseCurve, newApex);
  }
}

export class HalfSpace3D extends Transformable3D<HalfSpace3D> {
  constructor(
    public readonly plane: Plane,
    public readonly inverse: boolean,
  ) {
    super();
  }

  transform(transform: Transform3D): HalfSpace3D {
    const plane = this.plane.transform(transform);
    return new HalfSpace3D(plane, this.inverse);
  }

  slice(slicePlane: Plane): HalfPlane2D {
    const planeSlice = this.plane.slice(slicePlane);
    return new HalfPlane2D(planeSlice, this.inverse);
  }
}

export type Surface3D = Plane | Cone | Cylinder;

export type Solid = Cylinder | Cone | Partition3D;

export class Patch extends Transformable3D<Patch> {
  constructor(
    public readonly baseSurface: Surface3D | Solid,
    public readonly clipSections: HalfSpace3D[],
  ) {
    super();
  }

  clip(plane: Plane, reverse = false): Patch {
    const clipSection = new HalfSpace3D(plane, reverse);
    return new Patch(this.baseSurface, [...this.clipSections, clipSection]);
  }

  slice(slicePlane: Plane): Segment2D {
    const curveSlice = this.baseSurface.slice(slicePlane);
    const halfPlanes = this.clipSections.map((halfSpace) =>
      halfSpace.slice(slicePlane),
    );

    return new Segment2D(curveSlice, halfPlanes);
  }

  transform(transform: Transform3D): Patch {
    const baseSurface = this.baseSurface.transform(transform);
    const section = this.clipSections.map((halfSpace) =>
      halfSpace.transform(transform),
    );
    return new Patch(baseSurface, section);
  }
}

export class Partition3D extends Transformable3D<Partition3D> {
  constructor(
    public readonly baseSolid: Solid,
    public readonly clipSections: HalfSpace3D[],
  ) {
    super();
  }

  clip(plane: Plane, reverse = false): Patch {
    const clipSection = new HalfSpace3D(plane, reverse);
    return new Patch(this.baseSolid, [clipSection]);
  }

  partition(plane: Plane, reverse = false): Partition3D {
    const clipSection = new HalfSpace3D(plane, reverse);
    return new Partition3D(this.baseSolid, [...this.clipSections, clipSection]);
  }

  slice(slicePlane: Plane): Partition2D {
    const curveSlice = this.baseSolid.slice(slicePlane);
    const halfPlanes = this.clipSections.map((halfSpace) =>
      halfSpace.slice(slicePlane),
    );

    return new Partition2D(curveSlice, halfPlanes);
  }

  transform(transform: Transform3D): Partition3D {
    const baseSolid = this.baseSolid.transform(transform);
    const section = this.clipSections.map((halfSpace) =>
      halfSpace.transform(transform),
    );
    return new Partition3D(baseSolid, section);
  }
}
