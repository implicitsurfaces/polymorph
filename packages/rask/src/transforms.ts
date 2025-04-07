import { Angle } from "./angle";
import {
  diagonalMatrix4x4,
  IDENTITY_MATRIX_3x3,
  IDENTITY_MATRIX_4x4,
  Matrix3x3,
  Matrix4x4,
} from "./matrices";

export class Transform2D {
  constructor(
    public readonly matrix: Matrix3x3 = IDENTITY_MATRIX_3x3,
    private _inverseMatrix: Matrix3x3 | null = null,
  ) {}

  get repr(): string {
    return `Transform2D(${this.matrix.repr})`;
  }

  compose(other: Transform2D): Transform2D {
    return new Transform2D(this.matrix.mul(other.matrix));
  }

  followedBy(other: Transform2D): Transform2D {
    return other.compose(this);
  }

  precededBy(other: Transform2D): Transform2D {
    return this.compose(other);
  }

  reverse(): Transform2D {
    if (!this._inverseMatrix) {
      this._inverseMatrix = this.matrix.inverse();
    }
    return new Transform2D(this._inverseMatrix!);
  }
}

export const translationTransform2D = (x: number, y: number): Transform2D =>
  new Transform2D(
    new Matrix3x3(1, 0, x, 0, 1, y, 0, 0, 1),
    new Matrix3x3(1, 0, -x, 0, 1, -y, 0, 0, 1),
  );

export const rotationTransform2D = (angle: Angle): Transform2D =>
  new Transform2D(
    new Matrix3x3(
      angle.cos(),
      -angle.sin(),
      0,
      angle.sin(),
      angle.cos(),
      0,
      0,
      0,
      1,
    ),
    new Matrix3x3(
      angle.cos(),
      angle.sin(),
      0,
      -angle.sin(),
      angle.cos(),
      0,
      0,
      0,
      1,
    ),
  );

export const scalingTransform2D = (
  xFactor: number,
  yFactor: number,
): Transform2D =>
  new Transform2D(
    new Matrix3x3(xFactor, 0, 0, 0, yFactor, 0, 0, 0, 1),
    new Matrix3x3(1 / xFactor, 0, 0, 0, 1 / yFactor, 0, 0, 0, 1),
  );

export const rawTransform2D = (
  x11: number,
  x12: number,
  x13: number,
  x21: number,
  x22: number,
  x23: number,
  x31: number,
  x32: number,
  x33: number,
): Transform2D =>
  new Transform2D(new Matrix3x3(x11, x12, x13, x21, x22, x23, x31, x32, x33));

export const rotationAroundPointTransform2D = (
  x: number,
  y: number,
  angle: Angle,
): Transform2D => {
  return translationTransform2D(-x, -y)
    .followedBy(rotationTransform2D(angle))
    .followedBy(translationTransform2D(x, y));
};

interface Transformable2D {
  transform(transform: Transform2D): Transformable2D;
}

interface VecLike {
  x: number;
  y: number;
}

export function translate<T extends Transformable2D>(
  shape: T,
  vec: VecLike,
): T {
  return shape.transform(translationTransform2D(vec.x, vec.y)) as T;
}

export function rotate<T extends Transformable2D>(shape: T, angle: Angle): T {
  return shape.transform(rotationTransform2D(angle)) as T;
}

export function rotateAroundPoint<T extends Transformable2D>(
  shape: T,
  point: VecLike,
  angle: Angle,
): T {
  return shape.transform(
    rotationAroundPointTransform2D(point.x, point.y, angle),
  ) as T;
}

export function scale<T extends Transformable2D>(
  shape: T,
  xFactor: number,
  yFactor: number,
): T {
  return shape.transform(scalingTransform2D(xFactor, yFactor)) as T;
}

export class Transform3D {
  constructor(
    public readonly matrix: Matrix4x4 = IDENTITY_MATRIX_4x4,
    private _inverseMatrix: Matrix4x4 | null = null,
  ) {}

  get repr(): string {
    return `Transform3D(${this.matrix.repr})`;
  }

  compose(other: Transform3D): Transform3D {
    return new Transform3D(this.matrix.mul(other.matrix));
  }

  followedBy(other: Transform3D): Transform3D {
    return other.compose(this);
  }

  precededBy(other: Transform3D): Transform3D {
    return this.compose(other);
  }

  reverse(): Transform3D {
    if (!this._inverseMatrix) {
      this._inverseMatrix = this.matrix.inverse();
    }
    return new Transform3D(this._inverseMatrix!);
  }
}

export const translationTransform3D = (
  x: number,
  y: number,
  z: number,
): Transform3D =>
  new Transform3D(
    new Matrix4x4(1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z, 0, 0, 0, 1),
    new Matrix4x4(1, 0, 0, -x, 0, 1, 0, -y, 0, 0, 1, -z, 0, 0, 0, 1),
  );

export const scalingTransform3D = (
  xFactor: number,
  yFactor: number,
  zFactor: number,
): Transform3D =>
  new Transform3D(
    diagonalMatrix4x4(xFactor, yFactor, zFactor, 1),
    diagonalMatrix4x4(1 / xFactor, 1 / yFactor, 1 / zFactor, 1),
  );

export const rawTransform3D = (
  x11: number,
  x12: number,
  x13: number,
  x14: number,
  x21: number,
  x22: number,
  x23: number,
  x24: number,
  x31: number,
  x32: number,
  x33: number,
  x34: number,
  x41: number,
  x42: number,
  x43: number,
  x44: number,
): Transform3D =>
  new Transform3D(
    new Matrix4x4(
      x11,
      x12,
      x13,
      x14,
      x21,
      x22,
      x23,
      x24,
      x31,
      x32,
      x33,
      x34,
      x41,
      x42,
      x43,
      x44,
    ),
  );

export const rotationTransform3D = (
  angle: Angle,
  x: number,
  y: number,
  z: number,
): Transform3D => {
  // Normalize the axis vector
  const length = Math.sqrt(x * x + y * y + z * z);
  if (length !== 0) {
    x /= length;
    y /= length;
    z /= length;
  }

  const cosAngle = angle.cos();
  const sinAngle = angle.sin();
  const oneMinusCos = 1 - cosAngle;

  const matrix = new Matrix4x4(
    cosAngle + x * x * oneMinusCos,
    x * y * oneMinusCos - z * sinAngle,
    x * z * oneMinusCos + y * sinAngle,
    0,
    y * x * oneMinusCos + z * sinAngle,
    cosAngle + y * y * oneMinusCos,
    y * z * oneMinusCos - x * sinAngle,
    0,
    z * x * oneMinusCos - y * sinAngle,
    z * y * oneMinusCos + x * sinAngle,
    cosAngle + z * z * oneMinusCos,
    0,
    0,
    0,
    0,
    1,
  );

  return new Transform3D(matrix, matrix.transpose());
};

export const rotationAroundXAxisTransform3D = (angle: Angle): Transform3D => {
  // prettier-ignore
  const matrix = new Matrix4x4(
    1, 0, 0, 0,
    0, angle.cos(), -angle.sin(), 0,
    0, angle.sin(), angle.cos(), 0,
    0, 0, 0, 1,
  );
  return new Transform3D(matrix, matrix.transpose());
};

export const rotationAroundYAxisTransform3D = (angle: Angle): Transform3D => {
  // prettier-ignore
  const matrix = new Matrix4x4(
    angle.cos(), 0, angle.sin(), 0,
    0, 1, 0, 0,
    -angle.sin(), 0, angle.cos(), 0,
    0, 0, 0, 1,
  );
  return new Transform3D(matrix, matrix.transpose());
};

export const rotationAroundZAxisTransform3D = (angle: Angle): Transform3D => {
  // prettier-ignore
  const matrix = new Matrix4x4(
    angle.cos(), -angle.sin(), 0, 0,
    angle.sin(), angle.cos(), 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  );
  return new Transform3D(matrix, matrix.transpose());
};

interface Transformable3D {
  transform(transform: Transform3D): Transformable3D;
}

interface Vec3DLike {
  x: number;
  y: number;
  z: number;
}

export function translate3D<T extends Transformable3D>(
  shape: T,
  vec: Vec3DLike,
): T {
  return shape.transform(translationTransform3D(vec.x, vec.y, vec.z)) as T;
}

export function rotate3D<T extends Transformable3D>(
  shape: T,
  angle: Angle,
  axis: "x" | "y" | "z" = "x",
): T {
  if (axis === "x") {
    return shape.transform(rotationAroundXAxisTransform3D(angle)) as T;
  }

  if (axis === "y") {
    return shape.transform(rotationAroundYAxisTransform3D(angle)) as T;
  }

  if (axis === "z") {
    return shape.transform(rotationAroundZAxisTransform3D(angle)) as T;
  }

  throw new Error(`Invalid axis: ${axis}`);
}
