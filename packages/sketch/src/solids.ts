import { Angle, Vec2 } from "./geom";
import { Plane, Point3D, projectPoint, UnitVec3, Vec3 } from "./geom-3d";
import { Num, ONE, ZERO } from "./num";
import { hypot } from "./num-ops";
import { DistField, SolidDistField } from "./types";

export class Sphere implements SolidDistField {
  constructor(public radius: Num) {}

  valueAt(point: Point3D): Num {
    return point.vecFromOrigin().norm().sub(this.radius);
  }
}

export class Extrusion implements SolidDistField {
  constructor(
    public readonly height: Num,
    public readonly profile: DistField,
    public readonly plane: Plane,
  ) {}

  valueAt(point: Point3D): Num {
    const zVal = this.plane.zAxis
      .dot(point.vecFromOrigin())
      .abs()
      .sub(this.height.div(2));
    const xyVal = this.profile.distanceTo(projectPoint(point, this.plane));

    // This is non zero only  when both distance are negative (i.e. inside)
    // In that case we choose the closest distance (the max of two negative numbers)
    const insideDistance = xyVal.max(zVal).min(ZERO);

    // this is non zero only when at least one distance is positive (i.e. outside)
    // In that case we take the norm of the two distances as xy_dist is (x^2 + y^2)^0.5
    // norm(xy_distance, z_distance) is (x^2 + y^2 + z^2)^0.5
    const xyOutsideDistance = xyVal.max(ZERO);
    const zOutsideDistance = zVal.max(ZERO);
    const outsideDistance = hypot(xyOutsideDistance, zOutsideDistance);

    return insideDistance.add(outsideDistance);
  }
}

export class SolidRotation implements SolidDistField {
  constructor(
    public readonly angle: Angle,
    public readonly profile: SolidDistField,
    public readonly axis: UnitVec3,
  ) {}

  valueAt(point: Point3D): Num {
    const rotatedVec = point.vecFromOrigin().rotate(this.angle, this.axis);
    return this.profile.valueAt(rotatedVec.pointFromOrigin());
  }
}

export class SolidTranslation implements SolidDistField {
  constructor(
    public readonly translation: Vec3,
    public readonly solid: SolidDistField,
  ) {}

  valueAt(point: Point3D): Num {
    return this.solid.valueAt(point.sub(this.translation));
  }
}

export class Cone implements SolidDistField {
  constructor(
    public readonly radius: Num,
    public readonly height: Num,
  ) {}

  valueAt(point: Point3D): Num {
    /* We are using the circular symmetry around the center of the cone
     * to simplify the distance calculation.
     * We are in the coordiantes system of a side view of the cone (the
     * triangle).
     */
    const q = new Vec2(this.radius, this.height);

    const xyCoord = hypot(point.x, point.y);
    const pInTriangle = new Vec2(xyCoord, point.z);

    const hypotCoord = pInTriangle.dot(q).div(q.dot(q)).max(ZERO).min(ONE);
    const hypotPosition = pInTriangle.sub(q.scale(hypotCoord));

    const radiusCoord = xyCoord.div(this.radius).max(ZERO).min(ONE);
    const radiusPosition = pInTriangle.sub(
      new Vec2(this.radius.mul(radiusCoord), this.height),
    );

    const squareDistance = hypotPosition
      .dot(hypotPosition)
      .min(radiusPosition.dot(radiusPosition));

    const zSign = pInTriangle.y.sign();
    const hypotSign = pInTriangle.cross(q).mul(zSign);
    const radiusSign = pInTriangle.y.sub(this.height).mul(zSign);

    return squareDistance.sqrt().mul(hypotSign.max(radiusSign).sign());
  }
}

export class ConeSurface implements SolidDistField {
  constructor(
    public readonly radius: Num,
    public readonly height: Num,
  ) {}

  valueAt(point: Point3D): Num {
    /* We are using the circular symmetry around the center of the cone
     * to simplify the distance calculation.
     * We are in the coordiantes system of a side view of the cone (the
     * triangle).
     */
    const q = new Vec2(this.radius, this.height);

    const xyCoord = hypot(point.x, point.y);
    const pInTriangle = new Vec2(xyCoord, point.z);

    const hypotCoord = pInTriangle.dot(q).div(q.dot(q)).max(ZERO).min(ONE);
    const hypotPosition = pInTriangle.sub(q.scale(hypotCoord));

    return hypotPosition.norm();
  }
}
