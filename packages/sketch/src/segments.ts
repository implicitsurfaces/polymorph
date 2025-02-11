import {
  Angle,
  Point,
  SolidAngle,
  Vec2,
  arcTan,
  twoVectorsAngle,
} from "./geom";
import { candidateClosestPointsWithinEllipseArc } from "./geom-utils/closestPointOnEllipse";
import { Num } from "./num";
import { min, max, clamp, ifTruthyElse } from "./num-ops";
import { Segment } from "./types";

export class LineSegment implements Segment {
  private segment: Vec2;
  constructor(
    readonly p1: Point,
    readonly p2: Point,
  ) {
    this.segment = p1.vecTo(p2);
  }

  solidAngle(p: Point): SolidAngle {
    const a = this.p1.vecTo(p);
    const b = this.p2.vecTo(p);

    return new SolidAngle(0).addAngle(twoVectorsAngle(a, b));
  }

  distanceTo(p: Point): Num {
    const startToP = p.vecFrom(this.p1);
    const parametricPosition = startToP
      .dot(this.segment)
      .div(this.segment.dot(this.segment));

    const clampedPosition = clamp(parametricPosition, 0, 1);

    const projectedPoint = this.p1.add(this.segment.scale(clampedPosition));
    return p.vecFrom(projectedPoint).norm();
  }
}

export function arcWindingNumberIndefiniteIntegral(
  t: Angle,
  radius: Num,
  x: Num,
  y: Num,
): SolidAngle {
  /*
   * The general formula for the winding number of a circle is the integral of
   * the following function:
   *
   * w(x, y) = (x dy - y dx) / (x^2 + y^2)
   *
   * In the case of an arc of circle, we can express the parametric equation as:
   *
   * x = a + r cos(t)
   * y = b + r sin(t)
   *
   * Where (a, b) is the center of the circle and r is the radius. We can then
   * plug this into the winding number formula and integrate it with respect to t.
   *
   * With the help of wolfram alpha, we can find the following indefinite integral.
   *
   * This gives use the formula used here.
   */

  const halfT = t.half();
  const sinT = halfT.sin();
  const cosT = halfT.cos();

  const term1 = radius.mul(y).mul(2);
  const term2 = x.add(radius).square().add(y.square()).mul(sinT).div(cosT);

  const angleX = term1.sub(term2);
  const angleY = radius.square().sub(x.square()).sub(y.square());

  return new SolidAngle(0)
    .addAngle(arcTan(angleX, angleY))
    .add(new SolidAngle(0).addAngle(t).half());
}

export function arcWindingNumberAtPi(radius: Num, x: Num, y: Num) {
  const distToCenter = radius.square().sub(x.square().add(y.square()));
  return distToCenter.sign().add(1).div(2);
}

export class BulgingSegment implements Segment {
  constructor(
    readonly p1: Point,
    readonly p2: Point,
    readonly bulge: Num,
  ) {}

  get chord() {
    return this.p2.vecFrom(this.p1);
  }

  get center() {
    const bb = this.bulge.sub(this.bulge.inv()).div(4);
    return this.p1.midPoint(this.p2).sub(this.chord.perp().scale(bb));
  }

  get radius() {
    return this.chord.norm().div(4).mul(this.bulge.add(this.bulge.inv())).abs();
  }

  private get p1SortAngle() {
    return this.p1.vecFrom(this.center).asAngle().asSortValue();
  }

  private get p2SortAngle() {
    return this.p2.vecFrom(this.center).asAngle().asSortValue();
  }

  distanceTo(p: Point): Num {
    const inSectorDistance = p
      .vecFrom(this.center)
      .norm()
      .sub(this.radius)
      .abs();
    const outSectorDistance = min(
      p.vecFrom(this.p1).norm(),
      p.vecFrom(this.p2).norm(),
    );

    const orientation = this.bulge.sign();

    const pSortVal = orientation.mul(
      p.vecFrom(this.center).asAngle().asSortValue(),
    );
    const p1SortVal = orientation.mul(this.p1SortAngle);
    const p2SortVal = orientation.mul(this.p2SortAngle);

    /*
     * We want to decide if we return the inSectorDistance or the outSectorDistance
     * We can do this by comparing the angles of the start, end and the point
     *
     * The gist of it is that if the sort order of the points is a even permutation of
     * p1-point-p2 (i.e. the point is in the sector), we return the inSectorDistance
     * otherwise we return the outSectorDistance
     */

    const minSortVal = min(p1SortVal, p2SortVal, pSortVal);
    const maxSortVal = max(p1SortVal, p2SortVal, pSortVal);

    const p2IsExtrema = p2SortVal
      .equals(minSortVal)
      .or(p2SortVal.equals(maxSortVal));

    return ifTruthyElse(
      p1SortVal.lessThan(pSortVal),
      ifTruthyElse(p2IsExtrema, inSectorDistance, outSectorDistance),
      ifTruthyElse(p2IsExtrema, outSectorDistance, inSectorDistance),
    );
  }

  solidAngle(p: Point): SolidAngle {
    const startAngle = this.p1.vecFrom(this.center).asAngle();
    const endAngle = this.p2.vecFrom(this.center).asAngle();

    const pVec = p.vecFrom(this.center);

    const endAngleIntegral = arcWindingNumberIndefiniteIntegral(
      endAngle,
      this.radius,
      pVec.x,
      pVec.y,
    );
    const startAngleIntegral = arcWindingNumberIndefiniteIntegral(
      startAngle,
      this.radius,
      pVec.x,
      pVec.y,
    );

    /* First, we consider the case where the angles are oriented counter
     * clockwise
     *
     * We need to consider three cases: - both angles are smaller than pi
     * - both angles are greater than pi - one angle is smaller than pi and
     * the other is greater than pi
     *
     * If the angles are on the same side of pi, we cross the pi line if the
     * end angle is smaller than the start angle. This is the same for when
     * they are both bigger and small than pi - so we really only have two
     * cases, either the angles are on the same side of pi or not.
     *
     * We can use the sign of the product of the sin of the angles to check
     * if the angles are on the same side of pi.
     */

    const sameSideOfPiSign = startAngle.sin().mul(endAngle.sin());

    /* Then, as the cases are the inverse of each other, we can use the sign
     * of the product of the differences
     *
     * An angle is smaller than pi if the difference between the angle and
     * pi is positive and greater than pi if the difference is negative
     *
     * When the angles are on different sides of pi, we cross pi if the end
     * angles is bigger than the start angle. As this is the opposite of the
     * case where the angles are on the same side of pi, we can use the sign
     * of the product of the differences to determine if the angles are on
     * the same side of pi.
     */

    const spanSign = endAngle.asSortValue().sub(startAngle.asSortValue());

    /* We then need to take into account the case where the angles are
     * clockwise. In that case we need to make two changes:
     *
     * - the angles are inverted (i.e. the is_crossing_pi needs to invert
     * its sign)
     *
     * - the integral needs to be inverted (i.e. we need to subtract the
     * integral from the start to the end)
     */

    const orientationSign = this.bulge.sign();
    const isCrossingPi = sameSideOfPiSign
      .mul(spanSign)
      .mul(orientationSign)
      .lessThan(0);

    const piCrossingCorrectionTurns = ifTruthyElse(
      isCrossingPi,
      orientationSign.mul(arcWindingNumberAtPi(this.radius, pVec.x, pVec.y)),
      0,
    );

    const correctionSolidAngle = new SolidAngle(piCrossingCorrectionTurns);

    return endAngleIntegral.sub(startAngleIntegral).add(correctionSolidAngle);
  }
}

export class EllipseArcSegment implements Segment {
  readonly p1: Point;
  readonly p2: Point;
  constructor(
    readonly majorRadius: Num,
    readonly minorRadius: Num,
    readonly startAngle: Angle,
    readonly endAngle: Angle,
    readonly orientation: Num,
    readonly center: Point,
    readonly xAxisAngle: Angle,
    p1?: Point,
    p2?: Point,
  ) {
    this.p1 =
      p1 ??
      this.center.add(
        new Vec2(
          this.majorRadius.mul(this.startAngle.cos()),
          this.minorRadius.mul(this.startAngle.sin()),
        ).rotate(this.xAxisAngle),
      );

    this.p2 =
      p2 ??
      this.center.add(
        new Vec2(
          this.majorRadius.mul(this.endAngle.cos()),
          this.minorRadius.mul(this.endAngle.sin()),
        ).rotate(this.xAxisAngle),
      );
  }

  private pointInEllipseCoordinates(p: Point): Point {
    return p
      .vecFromOrigin()
      .sub(this.center.vecFromOrigin())
      .rotate(this.xAxisAngle.neg())
      .pointFromOrigin();
  }

  distanceTo(p: Point): Num {
    const point = this.pointInEllipseCoordinates(p);
    const candidates = candidateClosestPointsWithinEllipseArc(
      this.majorRadius,
      this.minorRadius,
      this.startAngle,
      this.endAngle,
      this.orientation,
      point,
    );

    const distances = candidates.map((closestPoint) =>
      point.vecFrom(closestPoint).norm(),
    );

    const minDist = min(...(distances as [Num]));

    const firstDist = p.vecFrom(this.p1).norm();
    const lastDist = p.vecFrom(this.p2).norm();

    return minDist.min(firstDist).min(lastDist);
  }

  solidAngle(p: Point): SolidAngle {
    return new SolidAngle(p.x);
  }
}
