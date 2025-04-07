import { diagonalMatrix4x4, Matrix3x3, Matrix4x4 } from "./matrices";

export function inPlaneProjectionMatrix(
  plane: [number, number, number, number],
): Matrix4x4 {
  return diagonalMatrix4x4(1, 1, 1, 2).sub(normalProjectionMatrix(plane));
}

export function normalProjectionMatrix(
  plane: [number, number, number, number],
): Matrix4x4 {
  const [a, b, c, d] = plane;

  const n = a * a + b * b + c * c;

  // prettier-ignore
  return new Matrix4x4(
    (a * a) / n, (a * b) / n, (a * c) / n, (d * a) / n,
    (b * a) / n, (b * b) / n, (b * c) / n, (d * b) / n,
    (c * a) / n, (c * b) / n, (c * c) / n, (d * c) / n,
    0, 0, 0, 1,
  );
}

export function asInPlaneTransformation(matrix: Matrix4x4): Matrix3x3 {
  // prettier-ignore
  return new Matrix3x3(
    matrix.x11, matrix.x12, matrix.x14,
    matrix.x21, matrix.x22, matrix.x24, 
    matrix.x41, matrix.x42, matrix.x44,
  );
}

export function projectToPlaneInDirection(
  plane: [number, number, number, number],
  direction: [number, number, number],
): Matrix4x4 {
  const [a, b, c, d] = plane;
  const [dx, dy, dz] = direction;

  // Compute denominator: n · d_direction
  const denom = a * dx + b * dy + c * dz;
  if (denom === 0) {
    throw new Error("Projection direction is parallel to the plane (n·d = 0)");
  }
  const invDenom = 1 / denom;

  return new Matrix4x4(
    1 - dx * a * invDenom,
    -dx * b * invDenom,
    -dx * c * invDenom,
    -d * dx * invDenom,
    -dy * a * invDenom,
    1 - dy * b * invDenom,
    -dy * c * invDenom,
    -d * dy * invDenom,
    -dz * a * invDenom,
    -dz * b * invDenom,
    1 - dz * c * invDenom,
    -d * dz * invDenom,
    0,
    0,
    0,
    1,
  );
}

export function projectToPlaneWithPoint(
  plane: [number, number, number, number],
  point: [number, number, number],
): Matrix4x4 {
  const [a, b, c, d] = plane;
  const [x0, y0, z0] = point;

  const dotNE = a * x0 + b * y0 + c * z0;
  const C = dotNE + d;

  // prettier-ignore
  return new Matrix4x4(
    x0 * a - C, x0 * b, x0 * c, d * x0,
    y0 * a, y0 * b - C, y0 * c, d * y0,
    z0 * a, z0 * b, z0 * c - C, d * z0,
    a, b, c, -dotNE,
  );
}
