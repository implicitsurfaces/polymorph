import { Angle, Point, Vec2 } from "../geom";
import { asNum, NEG_ONE, Num, ONE, ZERO } from "../num";
import { hypot, ifTruthyElse, max, min } from "../num-ops";

function pow3(x: Num) {
  return x.mul(x).mul(x);
}

export function applySign(from: Num, to: Num) {
  return ifTruthyElse(from.lessThan(ZERO), to.mul(NEG_ONE), to);
}

export function flipWhenAngleInOpposite(
  angle: Angle,
  vec2: Vec2,
  toFlip: Point,
) {
  const xSign = angle.cos().mul(vec2.x);
  const ySign = angle.sin().mul(vec2.y);
  const inQ3 = xSign.lessThan(ZERO).and(ySign.lessThan(ZERO));

  return new Point(
    ifTruthyElse(inQ3, toFlip.x.neg(), toFlip.x),
    ifTruthyElse(inQ3, toFlip.y.neg(), toFlip.y),
  );
}

function ifTruthyElseForAngles(
  condition: Num,
  ifNonZero: Angle,
  ifZero: Angle,
) {
  const outCos = ifTruthyElse(condition, ifNonZero.cos(), ifZero.cos());
  const outSin = ifTruthyElse(condition, ifNonZero.sin(), ifZero.sin());

  return new Angle(outCos, outSin);
}

export function clampAngle(
  angle: Angle,
  p1: Angle,
  p2: Angle,
  orientation: Num,
) {
  const withinRange = angleWithinRange(angle, p1, p2, orientation);

  return ifTruthyElseForAngles(withinRange, angle, p1);
}

export function angleWithinRange(
  angle: Angle,
  p1: Angle,
  p2: Angle,
  orientation: Num,
) {
  const pSortVal = orientation.mul(angle.asSortValue());
  const p1SortVal = orientation.mul(p1.asSortValue());
  const p2SortVal = orientation.mul(p2.asSortValue());

  /*
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
    ifTruthyElse(p2IsExtrema, ONE, ZERO),
    ifTruthyElse(p2IsExtrema, ZERO, ONE),
  );
}

export const T0 = asNum(0.707);

export function closestPointOnEllipse(
  majorRadius: Num,
  minorRadius: Num,
  point: Point,
) {
  const px = point.x;
  const py = point.y;

  const startAngle = new Angle(applySign(point.x, T0), applySign(point.y, T0));

  let angle = startAngle;

  const majorRadiusSq = majorRadius.square();
  const minorRadiusSq = minorRadius.square();

  const squareDiff = majorRadiusSq.sub(minorRadiusSq);

  const evoluteX = squareDiff.div(majorRadius);
  const evoluteY = squareDiff.neg().div(minorRadius);

  for (let i = 0; i < 3; i++) {
    const x = majorRadius.mul(angle.cos());
    const y = minorRadius.mul(angle.sin());

    const ex = evoluteX.mul(pow3(angle.cos()));
    const ey = evoluteY.mul(pow3(angle.sin()));

    const rx = x.sub(ex);
    const ry = y.sub(ey);

    const qx = px.sub(ex);
    const qy = py.sub(ey);

    const r = hypot(rx, ry);
    const q = hypot(qx, qy);

    const tx = qx.mul(r).div(q).add(ex).div(majorRadius).max(-1).min(1);
    const ty = qy.mul(r).div(q).add(ey).div(minorRadius).max(-1).min(1);

    const t = hypot(tx, ty);

    angle = new Angle(tx.div(t), ty.div(t));
  }

  return new Point(majorRadius.mul(angle.cos()), minorRadius.mul(angle.sin()));
}

function closestAngleOnEllipseArcFromAngle(
  majorRadius: Num,
  minorRadius: Num,
  fromAngle: Angle,
  minCos: Num,
  maxCos: Num,
  minSin: Num,
  maxSin: Num,
  point: Point,
) {
  let angle = fromAngle;
  const majorRadiusSq = majorRadius.square();
  const minorRadiusSq = minorRadius.square();

  const squareDiff = majorRadiusSq.sub(minorRadiusSq);

  const evoluteX = squareDiff.div(majorRadius);
  const evoluteY = squareDiff.neg().div(minorRadius);

  for (let i = 0; i < 5; i++) {
    const x = majorRadius.mul(angle.cos());
    const y = minorRadius.mul(angle.sin());

    const p = new Point(x, y);

    const ex = evoluteX.mul(pow3(angle.cos()));
    const ey = evoluteY.mul(pow3(angle.sin()));

    const e = new Point(ex, ey);

    const r = e.vecTo(p);
    const q = e.vecTo(point);

    const c = e.add(q.normalize().scale(r.norm()));

    let tx = c.x.div(majorRadius).max(minCos).min(maxCos);
    const ty = c.y.div(minorRadius).max(minSin).min(maxSin);

    tx = ifTruthyElse(tx.or(ty), tx, minCos.or(maxCos));

    const t = hypot(tx, ty);

    angle = new Angle(tx.div(t), ty.div(t));
  }

  return angle;
}

const DEFAULT_ANGLES: [Angle, Num, Num, Num, Num][] = [
  [new Angle(T0, T0), ZERO, ONE, ZERO, ONE],
  [new Angle(T0.neg(), T0.neg()), NEG_ONE, ZERO, NEG_ONE, ZERO],
  [new Angle(T0.neg(), T0), NEG_ONE, ZERO, ZERO, ONE],
  [new Angle(T0, T0.neg()), ZERO, ONE, NEG_ONE, ZERO],
];

export function closestPointsOnEllipseArc(
  majorRadius: Num,
  minorRadius: Num,
  startAngle: Angle,
  endAngle: Angle,
  orientation: Num,
  point: Point,
): [Point, Point, Point, Point] {
  return DEFAULT_ANGLES.map(([angle, minCos, maxCos, minSin, maxSin]) =>
    closestAngleOnEllipseArcFromAngle(
      majorRadius,
      minorRadius,
      angle,
      minCos,
      maxCos,
      minSin,
      maxSin,
      point,
    ),
  )
    .map((angle) => clampAngle(angle, startAngle, endAngle, orientation))
    .map(
      (angle) =>
        new Point(majorRadius.mul(angle.cos()), minorRadius.mul(angle.sin())),
    ) as [Point, Point, Point, Point];
}
