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

export function canonicalCoordinateSystem([a, b, c, d]: [
  number,
  number,
  number,
  number,
]): Matrix4x4 {
  const magnitudeSquare = a ** 2 + b ** 2 + c ** 2;
  const magnitude = Math.sqrt(magnitudeSquare);

  const px = (a * d) / magnitudeSquare;
  const py = (b * d) / magnitudeSquare;
  const pz = (c * d) / magnitudeSquare;

  const n: V = [a / magnitude, b / magnitude, c / magnitude];

  const r: V = Math.abs(n[0]) < 0.9 ? [1, 0, 0] : [0, 0, 1];
  const u = normalize(gramSchmidt(r, n));
  const v = cross(n, u);

  // rotate the translation part
  const p = [
    px * u[0] + py * v[0] + pz * n[0],
    px * u[1] + py * v[1] + pz * n[1],
    px * u[2] + py * v[2] + pz * n[2],
  ];

  // prettier-ignore
  return new Matrix4x4(
    u[0], v[0], n[0], p[0],
    u[1], v[1], n[1], p[1],
    u[2], v[2], n[2], p[2],
    0, 0, 0, 1,
  );
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
  private _localToGlobalConversion: Matrix4x4 | null = null;
  constructor(public readonly globalToLocalConversion: Matrix4x4) {}

  transformInPlanneCoordinateSystem(transform: Transform3D): Matrix4x4 {
    return this.globalToLocalConversion.mul(
      transform.matrix.mul(this.localToGlobalConversion),
    );
  }

  inPlaneTransformMatrix(transform: Transform3D): Matrix4x4 {
    const baseMatrix = this.transformInPlanneCoordinateSystem(transform);
    // prettier-ignore
    return new Matrix4x4(
      baseMatrix.x11, baseMatrix.x12, 0, baseMatrix.x14,
      baseMatrix.x21, baseMatrix.x22, 0, baseMatrix.x24, 
      0, 0, 1, 0,
      0, 0, 0, 1,
    );
  }

  transform(transform: Transform3D): PlanarCoordinateSystem {
    return new PlanarCoordinateSystem(
      transform.reverse().matrix.mul(this.globalToLocalConversion),
    );
  }

  get localToGlobalConversion(): Matrix4x4 {
    if (!this._localToGlobalConversion) {
      const mat = this.globalToLocalConversion.inverse();
      this._localToGlobalConversion = mat.scale(1 / mat.x44);
    }
    return this._localToGlobalConversion;
  }

  globalToLocal(global: Vec3DLike): Vec2DLike {
    const colVec = new ColVec4(global.x, global.y, global.z, 1);
    const result = this.globalToLocalConversion.product(colVec);
    return { x: result.x1, y: result.x2 };
  }

  localToGlobal(local: Vec2DLike): Vec3DLike {
    const colVec = new ColVec4(local.x, local.y, 0, 1);
    const result = this.localToGlobalConversion.product(colVec);
    return { x: result.x1, y: result.x2, z: result.x3 };
  }
}
