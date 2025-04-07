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
    d * dx * invDenom,
    -dy * a * invDenom,
    1 - dy * b * invDenom,
    -dy * c * invDenom,
    d * dy * invDenom,
    -dz * a * invDenom,
    -dz * b * invDenom,
    1 - dz * c * invDenom,
    d * dz * invDenom,
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

  const dot = a * x0 + b * y0 + c * z0 + d;
  const invDot = 1 / dot;

  if (dot === 0) {
    throw new Error(
      "The point lies on the plane or is at infinity — can't project.",
    );
  }

  // Outer product of point and plane
  return new Matrix4x4(
    1 - x0 * a * invDot,
    x0 * b * invDot,
    x0 * c * invDot,
    x0 * d * invDot,
    y0 * a * invDot,
    1 - y0 * b * invDot,
    y0 * c * invDot,
    y0 * d * invDot,
    z0 * a * invDot,
    z0 * b * invDot,
    1 - z0 * c * invDot,
    z0 * d * invDot,
    a * invDot,
    b * invDot,
    c * invDot,
    1 - d * invDot,
  ).transpose();
}
