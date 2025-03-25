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

export const rawTransform = (
  x11: Num,
  x12: Num,
  x13: Num,
  x21: Num,
  x22: Num,
  x23: Num,
  x31: Num,
  x32: Num,
  x33: Num,
): Transform =>
  new Transform(new Matrix3x3(x11, x12, x13, x21, x22, x23, x31, x32, x33));

export const rotationAroundPointTransform = (
  angle: Angle,
  point: Point,
): Transform => {
  const p = point.vecFromOrigin();

  return translationTransform(p.neg())
    .followedBy(rotationTransform(angle))
    .followedBy(translationTransform(p));
};

export function alignDirectionRotationMatrix(
  fromDirection: UnitVec3,
  toDirection: UnitVec3,
) {
  const axis = fromDirection.cross(toDirection);
  const cosAngle = fromDirection.dot(toDirection);

  // TODO: handle the case when the directions are colinear

  const k = ONE.sub(cosAngle).div(ONE.sub(cosAngle.square()));

  return new Matrix3x3(
    k.mul(axis.x.square()).add(cosAngle),
    k.mul(axis.x).mul(axis.y).sub(axis.z),
    k.mul(axis.x).mul(axis.z).add(axis.y),
    k.mul(axis.x).mul(axis.y).add(axis.z),
    k.mul(axis.y.square()).add(cosAngle),
    k.mul(axis.y).mul(axis.z).sub(axis.x),
    k.mul(axis.x).mul(axis.z).sub(axis.y),
    k.mul(axis.y).mul(axis.z).add(axis.x),
    k.mul(axis.z.square()).add(cosAngle),
  );
}

function translationMatrix3D(v: Vec3) {
  return new Matrix3x3(ONE, ZERO, ZERO, ZERO, ONE, ZERO, v.x, v.y, v.z);
}

function liftTo3DMatrix(plane: Plane) {
  return new Matrix3x3(
    plane.xAxis.x,
    plane.yAxis.x,
    plane.origin.x,
    plane.xAxis.y,
    plane.yAxis.y,
    plane.origin.y,
    plane.xAxis.z,
    plane.yAxis.z,
    plane.origin.z,
  );
}

export function centerProjectionTransform(
  originPlane: Plane,
  targetPlane: Plane,
  projectionCenter: Point3D,
): Transform {
  const embed = liftTo3DMatrix(originPlane);
  const planeRotation = alignDirectionRotationMatrix(
    originPlane.zAxis,
    targetPlane.zAxis,
  );
  const centerTranslation = translationMatrix3D(
    projectionCenter.vecFromOrigin().neg(),
  );

  return new Transform(embed.mul(planeRotation).mul(centerTranslation));
}
