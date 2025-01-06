import { Angle } from "./geom";
import { Plane, Point3D, projectPoint, UnitVec3, Vec3 } from "./geom-3d";
import { Num, ZERO } from "./num";
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
