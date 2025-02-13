import { Angle, Vec2, ORIGIN as ORIGIN2D } from "./geom";
import {
  intersectLinePlane,
  Plane,
  Point3D,
  projectPoint,
  UnitVec3,
  Vec3,
  ORIGIN,
  Y_AXIS,
  embedPoint,
  distanceToLine,
  Z_AXIS,
} from "./geom-3d";
import { closestPointOnEllipse } from "./geom-utils/closestPointOnEllipse";
import { Num, ONE, ZERO } from "./num";
import { hypot } from "./num-ops";
import { conjugateDiametersEllipse } from "./segments-helpers";
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

export class EllipticCone implements SolidDistField {
  private readonly a: Point3D;
  private readonly b: Point3D;
  private readonly center: Point3D;

  constructor(
    public readonly majorRadius: Num,
    public readonly minorRadius: Num,
    public readonly height: Num,
  ) {
    this.a = new Point3D(this.majorRadius, ZERO, this.height);
    this.b = new Point3D(ZERO, this.minorRadius, this.height);
    this.center = new Point3D(ZERO, ZERO, this.height);
  }

  valueAt(point: Point3D): Num {
    const projectionPlaneNormal = Z_AXIS; //point.vecFromOrigin().normalize();

    const projectionPlane = new Plane(
      point,
      projectionPlaneNormal,
      Y_AXIS.cross(projectionPlaneNormal),
    );

    // We project the ellipse on the plane defined by the point
    const centerPrime = intersectLinePlane(
      this.center,
      this.center.vecFromOrigin(),
      projectionPlane,
    );
    const aPrime = intersectLinePlane(
      this.a,
      this.a.vecFromOrigin(),
      projectionPlane,
    );
    const bPrime = intersectLinePlane(
      this.b,
      this.b.vecFromOrigin(),
      projectionPlane,
    );

    const projectedC = projectPoint(centerPrime, projectionPlane);
    const projectedA = projectPoint(aPrime, projectionPlane);
    const projectedB = projectPoint(bPrime, projectionPlane);

    const projectedEllipse = conjugateDiametersEllipse(
      projectedC,
      projectedA,
      projectedB,
    );

    const closestPoint = closestPointOnEllipse(
      projectedEllipse.majorRadius,
      projectedEllipse.minorRadius,
      projectedEllipse.pointInEllipseCoordinates(ORIGIN2D), // the plane is defined such that its origin is the point
    );

    const embeddedPoint = embedPoint(closestPoint, projectionPlane);
    return distanceToLine(
      ORIGIN,
      embeddedPoint.vecFromOrigin().normalize(),
      point,
    ).mul(projectedEllipse.occupancySign(ORIGIN2D));
  }
}
