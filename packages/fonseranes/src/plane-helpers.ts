import { ColVec4, Matrix4x4 } from "./matrices";
import { Transform3D } from "./transforms";

type V = [number, number, number];

function cross(a: V, b: V): V {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function dot(a: V, b: V): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function normalize(v: V): V {
  const mag = Math.sqrt(dot(v, v));
  return [v[0] / mag, v[1] / mag, v[2] / mag];
}

function gramSchmidt(r: V, axis: V): V {
  const dotProduct = dot(r, axis);
  const projection = [
    axis[0] * dotProduct,
    axis[1] * dotProduct,
    axis[2] * dotProduct,
  ];
  return [r[0] - projection[0], r[1] - projection[1], r[2] - projection[2]];
}

export function planeCoordinateTransform([a, b, c, d]: [
  number,
  number,
  number,
  number,
]): Transform3D {
  const magnitudeSquare = a ** 2 + b ** 2 + c ** 2;
  const magnitude = Math.sqrt(magnitudeSquare);

  const px = -(a * d) / magnitudeSquare;
  const py = -(b * d) / magnitudeSquare;
  const pz = -(c * d) / magnitudeSquare;

  const n: V = [a / magnitude, b / magnitude, c / magnitude];

  const r: V = Math.abs(n[0]) < 0.9 ? [1, 0, 0] : [0, 0, 1];
  const u = normalize(gramSchmidt(r, n));
  const v = cross(n, u);

  const p: V = [px, py, pz];

  // prettier-ignore
  const localToGlobal = new Matrix4x4(
    u[0], v[0], n[0], p[0],
    u[1], v[1], n[1], p[1],
    u[2], v[2], n[2], p[2],
    0, 0, 0, 1,
  );

  const globalToLocal = new Matrix4x4(
    u[0],
    u[1],
    u[2],
    -dot(u, p),
    v[0],
    v[1],
    v[2],
    -dot(v, p),
    n[0],
    n[1],
    n[2],
    -dot(n, p),
    0,
    0,
    0,
    1,
  );

  return new Transform3D(localToGlobal, globalToLocal);
}

interface Vec2DLike {
  x: number;
  y: number;
}

interface Vec3DLike {
  x: number;
  y: number;
  z: number;
}

export class PlanarCoordinateSystem {
  constructor(public readonly conversion: Transform3D) {}

  transformInPlanneCoordinateSystem(transform: Transform3D): Matrix4x4 {
    return this.conversion.reverseMatrix.mul(
      transform.matrix.mul(this.conversion.matrix),
    );
  }

  get origin(): Vec3DLike {
    return {
      x: this.conversion.matrix.x14,
      y: this.conversion.matrix.x24,
      z: this.conversion.matrix.x34,
    };
  }

  get xAxis(): Vec3DLike {
    return {
      x: this.conversion.matrix.x11,
      y: this.conversion.matrix.x21,
      z: this.conversion.matrix.x31,
    };
  }

  get yAxis(): Vec3DLike {
    return {
      x: this.conversion.matrix.x12,
      y: this.conversion.matrix.x22,
      z: this.conversion.matrix.x32,
    };
  }

  get normal(): Vec3DLike {
    return {
      x: this.conversion.matrix.x13,
      y: this.conversion.matrix.x23,
      z: this.conversion.matrix.x33,
    };
  }

  get localToGlobalMatrix(): Matrix4x4 {
    return this.conversion.matrix;
  }

  get globalToLocalMatrix(): Matrix4x4 {
    return this.conversion.reverseMatrix;
  }

  transform(transform: Transform3D): PlanarCoordinateSystem {
    return new PlanarCoordinateSystem(this.conversion.compose(transform));
  }

  globalToLocal(global: Vec3DLike): Vec2DLike {
    const colVec = new ColVec4(global.x, global.y, global.z, 1);
    const result = this.conversion.reverseMatrix.product(colVec);
    return { x: result.x1, y: result.x2 };
  }

  localToGlobal(local: Vec2DLike): Vec3DLike {
    const colVec = new ColVec4(local.x, local.y, 0, 1);
    const result = this.conversion.matrix.product(colVec);
    return { x: result.x1, y: result.x2, z: result.x3 };
  }
}

export function canonicalCoordinateSystem([a, b, c, d]: [
  number,
  number,
  number,
  number,
]): PlanarCoordinateSystem {
  return new PlanarCoordinateSystem(planeCoordinateTransform([a, b, c, d]));
}
