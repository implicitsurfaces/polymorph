import {
  ColVec3,
  ColVec4,
  diagonalMatrix3x3,
  diagonalMatrix4x4,
  IDENTITY_MATRIX_4x4,
  Matrix3x3,
  Matrix3x4,
  Matrix4x3,
  Matrix4x4,
  RowVec4,
} from "./matrices";
import {
  canonicalCoordinateSystem,
  PlanarCoordinateSystem,
} from "./plane-helpers";
import { projectionTransform2D, Transform2D, Transform3D } from "./transforms";
import {
  ADD_Z_MATRIX,
  DROP_Z_MATRIX,
  normalProjectionMatrix,
  inPlaneProjectionMatrix,
  asInPlaneTransformation,
  projectToPlaneInDirection,
  projectToPlaneWithPoint,
} from "./transforms-helpers";

export class Primitive2D {
  constructor() {}
}

export class Point2D extends Primitive2D {
  constructor(
    public readonly x: number,
    public readonly y: number,
  ) {
    super();
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

export class Vec2D extends Primitive2D {
  constructor(
    public readonly x: number,
    public readonly y: number,
  ) {
    super();
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

  transform(transform: Transform2D): Vec2D {
    const colVec = new ColVec3(this.x, this.y, 0);
    const result = transform.matrix.product(colVec);
    if (result.x3 !== 0) {
      throw new Error("Invalid transformation: result is a point");
    }
    return new Vec2D(result.x1, result.x2);
  }
}

export class Line2D extends Primitive2D {
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
    return new Line2D(a, b, c);
  }

  static fromPointAndVec(p: Point2D, v: Vec2D): Line2D {
    const a = v.y;
    const b = -v.x;
    const c = a * p.x + b * p.y;
    return new Line2D(a, b, c);
  }

  levelSet(p: Point2D): number {
    return this.a * p.x + this.b * p.y - this.c;
  }

  transform(transform: Transform2D): Line2D {
    const matrix = transform.matrix.inverse().transpose();
    const colVec = new ColVec3(this.a, this.b, this.c);
    const result = matrix.product(colVec);
    return new Line2D(result.x1, result.x2, result.x3);
  }
}

export class HalfPlane2D extends Primitive2D {
  constructor(
    public readonly line: Line2D,
    public readonly greaterSide: boolean,
  ) {
    super();
  }

  levelSet(p: Point2D): number {
    const levelSet = this.line.levelSet(p);
    const isInside = this.greaterSide ? levelSet >= 0 : levelSet <= 0;
    return isInside ? 0 : levelSet;
  }

  transform(transform: Transform2D): HalfPlane2D {
    const line = this.line.transform(transform);
    return new HalfPlane2D(line, this.greaterSide);
  }
}

const Q_MAT = diagonalMatrix3x3(1, 1, -1);

export class Conic2D extends Primitive2D {
  private _matrix: Matrix3x3 | null = null;
  constructor(public readonly transformation: Transform2D) {
    super();
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

  transform(transform: Transform2D): Conic2D {
    const newTransform = this.transformation.compose(transform.reverse());
    return new Conic2D(newTransform);
  }
}

type Curve2D = Line2D | Conic2D;

export class Segment2D extends Primitive2D {
  constructor(
    public readonly curve: Curve2D,
    public readonly section: HalfPlane2D[],
  ) {
    super();
  }

  transform(transform: Transform2D): Segment2D {
    const curve = this.curve.transform(transform);
    const section = this.section.map((halfPlane) =>
      halfPlane.transform(transform),
    );
    return new Segment2D(curve, section);
  }

  levelSet(p: Point2D): number {
    const level = Math.abs(this.curve.levelSet(p));
    const boundaryLevelSet = this.section.map((halfPlane) =>
      Math.abs(halfPlane.levelSet(p)),
    );

    return Math.max(level, ...boundaryLevelSet);
  }
}

export class Primitive3D {
  constructor() {}
}

export class Point3D extends Primitive3D {
  constructor(
    public readonly x: number,
    public readonly y: number,
    public readonly z: number,
  ) {
    super();
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
}

export class Vec3D extends Primitive3D {
  constructor(
    public readonly x: number,
    public readonly y: number,
    public readonly z: number,
  ) {
    super();
  }

  scale(scalar: number): Vec3D {
    return new Vec3D(this.x * scalar, this.y * scalar, this.z * scalar);
  }

  magnitude(): number {
    return Math.sqrt(this.x ** 2 + this.y ** 2 + this.z ** 2);
  }
}

export class Plane extends Primitive3D {
  _coordinateSystem: PlanarCoordinateSystem | null = null;
  constructor(
    public readonly a: number,
    public readonly b: number,
    public readonly c: number,
    public readonly d: number,
  ) {
    super();
  }

  fromPointAndNormal(point: Point3D, normal: Vec3D): Plane {
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

  levelSet(point: Point3D): number {
    return this.a * point.x + this.b * point.y + this.c * point.z + this.d;
  }

  pointInPlane(point: Point3D): boolean {
    const levelSet = this.levelSet(point);
    return Math.abs(levelSet) < 1e-10;
  }

  get coordinateSystem(): PlanarCoordinateSystem {
    if (!this._coordinateSystem) {
      this._coordinateSystem = new PlanarCoordinateSystem(
        canonicalCoordinateSystem(this.asTuple()),
      );
    }
    return this._coordinateSystem!;
  }
}

function planarTransfrom(
  fromPlane: Plane,
  toPlane: Plane,
  transformMatrix3D: Matrix4x4,
): Transform2D {
  const fromCoords = fromPlane.coordinateSystem.globalToLocalConversion;
  const toCoords = toPlane.coordinateSystem.globalToLocalConversion;

  const transformationMatrix = toCoords.mul(transformMatrix3D).mul(fromCoords);

  return new Transform2D(asInPlaneTransformation(transformationMatrix));
}

export class Curve3D extends Primitive3D {
  constructor(
    public readonly curve: Curve2D,
    public readonly plane: Plane,
  ) {
    super();
  }

  projectIntoPlane(targetPlane: Plane): Curve3D {
    const transformMatrix =
      targetPlane.coordinateSystem.globalToLocalConversion.mul(
        this.plane.coordinateSystem.localToGlobalConversion,
      );
    const transform = new Transform2D(asInPlaneTransformation(transformMatrix));

    const newCurve = this.curve.transform(transform);
    return new Curve3D(newCurve, targetPlane);
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

export class Cylinder extends Primitive3D {
  constructor(public readonly baseCurve: Curve3D) {
    super();
  }

  slice(slicePlane: Plane): Curve3D {
    const projectionDir: [number, number, number] = [
      this.baseCurve.plane.a,
      this.baseCurve.plane.b,
      this.baseCurve.plane.c,
    ];
    const projectionMatrix = projectToPlaneInDirection(
      slicePlane.asTuple(),
      projectionDir,
    );

    const transform = planarTransfrom(
      this.baseCurve.plane,
      slicePlane,
      projectionMatrix,
    );

    const newCurve = this.baseCurve.curve.transform(transform);
    return new Curve3D(newCurve, slicePlane);
  }

  transform(transform: Transform3D): Cylinder {
    const newBaseCurve = this.baseCurve.transform(transform);
    return new Cylinder(newBaseCurve);
  }
}

export class Cone extends Primitive3D {
  constructor(
    public readonly baseCurve: Curve3D,
    public readonly apex: Point3D,
  ) {
    super();
  }

  slice(slicePlane: Plane): Curve3D {
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

    const newCurve = this.baseCurve.curve.transform(transform);
    return new Curve3D(newCurve, slicePlane);
  }

  transform(transform: Transform3D): Cone {
    const newBaseCurve = this.baseCurve.transform(transform);
    const newApex = this.apex.transform(transform);
    console.log("newApex", newApex);
    return new Cone(newBaseCurve, newApex);
  }
}

export class HalfSpace3D extends Primitive3D {
  constructor(
    public readonly plane: Plane,
    public readonly greaterSide: boolean,
  ) {
    super();
  }
}

type Surface3D = Plane | Cone | Cylinder;

export class Patch extends Primitive3D {
  constructor(
    public readonly baseSurface: Surface3D,
    public readonly section: HalfSpace3D[],
  ) {
    super();
  }
}
