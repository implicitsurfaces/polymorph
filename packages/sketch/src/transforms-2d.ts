import { Angle, Point, Vec2 } from "./geom";
import { Plane, Point3D, UnitVec3, Vec3 } from "./geom-3d";
import { ColVec3, IDENTITY_MATRIX_3x3, Matrix3x3 } from "./geom-utils/matrices";
import { Num, ONE, ZERO } from "./num";

export class Transform {
  constructor(public readonly matrix: Matrix3x3 = IDENTITY_MATRIX_3x3) {}

  compose(other: Transform): Transform {
    return new Transform(this.matrix.mul(other.matrix));
  }

  followedBy(other: Transform): Transform {
    return other.compose(this);
  }

  precededBy(other: Transform): Transform {
    return this.compose(other);
  }

  reverse(): Transform {
    return new Transform(this.matrix.inverse());
  }

  apply(point: Point): Point {
    const v = new ColVec3(point.x, point.y, ONE);
    const result = this.matrix.product(v);
    return new Point(result.x1, result.x2);
  }
}

export const translationTransform = (v: Vec2): Transform =>
  new Transform(new Matrix3x3(ONE, ZERO, v.x, ZERO, ONE, v.y, ZERO, ZERO, ONE));

export const rotationTransform = (angle: Angle): Transform =>
  new Transform(
    new Matrix3x3(
      angle.cos(),
      angle.sin().neg(),
      ZERO,
      angle.sin(),
      angle.cos(),
      ZERO,
      ZERO,
      ZERO,
      ONE,
    ),
  );

export const scalingTransform = (xFactor: Num, yFactor: Num): Transform =>
  new Transform(
    new Matrix3x3(xFactor, ZERO, ZERO, ZERO, yFactor, ZERO, ZERO, ZERO, ONE),
  );

export const rotationAroundPointTransform = (
  angle: Angle,
  point: Point,
): Transform => {
  const p = point.vecFromOrigin();

  return translationTransform(p.neg())
    .followedBy(rotationTransform(angle))
    .followedBy(translationTransform(p));
};
