import { Angle, Point, QUARTER_TURN, Vec2 } from "./geom";
import { NEG_ONE, Num, ONE } from "./num";
import { ifTruthyElse } from "./num-ops";
import { GenericEllipse } from "./profiles";
import { BulgingSegment, EllipseArcSegment } from "./segments";

function bulgeFromSandC(s: Num, c: Num) {
  return s.neg().div(c.add(c.square().add(s.square()).sqrt()));
}

export function bulgingSegmentUsingStartTangent(
  p1: Point,
  p2: Point,
  startTangent: Angle,
) {
  const chord = p1.vecTo(p2);
  const tgt = startTangent.asVec();

  const s = chord.cross(tgt);
  const c = chord.dot(tgt);

  return new BulgingSegment(p1, p2, bulgeFromSandC(s, c));
}

export function bulgingSegmentUsingStartControl(
  p1: Point,
  p2: Point,
  control: Point,
) {
  const chord = p1.vecTo(p2);
  const tgt = p1.vecTo(control).normalize();

  const s = chord.cross(tgt);
  const c = chord.dot(tgt);
  return new BulgingSegment(p1, p2, bulgeFromSandC(s, c));
}

export function bulgingSegmentUsingEndTangent(
  p1: Point,
  p2: Point,
  endTangent: Angle,
) {
  const chord = p1.vecTo(p2);
  const tgt = endTangent.asVec();

  const s = tgt.cross(chord);
  const c = tgt.dot(chord);
  return new BulgingSegment(p1, p2, bulgeFromSandC(s, c));
}

export function bulgingSegmentUsingEndControl(
  p1: Point,
  p2: Point,
  control: Point,
) {
  const chord = p1.vecTo(p2);
  const tgt = p2.vecTo(control).normalize();

  const s = tgt.cross(chord);
  const c = tgt.dot(chord);
  return new BulgingSegment(p1, p2, bulgeFromSandC(s, c));
}

export function threePointsBulgingSegment(p1: Point, p2: Point, p3: Point) {
  const chord0 = p1.vecTo(p3);
  const chord1 = p3.vecTo(p2);

  const s = chord1.cross(chord0);
  const c = chord1.dot(chord0);

  const bulge = s.neg().div(c.add(c.square().add(s.square()).sqrt()));
  return new BulgingSegment(p1, p2, bulge);
}

export function lineLineIntersection(p0: Point, v0: Vec2, p1: Point, v1: Vec2) {
  const crossDir = v0.cross(v1);
  const diffPoint = p1.vecFrom(p0);

  const param = diffPoint.cross(v1).div(crossDir);

  return p0.add(v0.scale(param));
}

export function biarcC(p1: Point, p2: Point, control: Point) {
  const w0 = control.vecFrom(p2).norm();
  const w1 = control.vecFrom(p1).norm();
  const w2 = p2.vecFrom(p1).norm();

  const perimeter = w0.add(w1).add(w2);

  const junction = p1
    .vecFromOrigin()
    .scale(w0)
    .add(p2.vecFromOrigin().scale(w1))
    .add(control.vecFromOrigin().scale(w2))
    .div(perimeter)
    .pointFromOrigin();

  const theta1 = control.vecFrom(p1).asAngle();
  const theta2 = control.vecFrom(p2).asAngle().opposite();

  return [
    bulgingSegmentUsingStartTangent(p1, junction, theta1),
    bulgingSegmentUsingEndTangent(junction, p2, theta2),
  ];
}

export function biarcLocusCenter(
  p1: Point,
  theta1: Angle,
  p2: Point,
  theta2: Angle,
) {
  const u1 = theta1.asVec();
  const u2 = theta2.asVec();

  const m1 = p2.midPoint(p1);
  const v1 = p2.vecFrom(p1).perp();

  const p3 = p1.add(u1);
  const p4 = p2.add(u2);

  const m2 = p3.midPoint(p4);
  const v2 = p4.vecFrom(p3).perp();

  return lineLineIntersection(m1, v1, m2, v2);
}

export function biarcS(p1: Point, p2: Point, control1: Point, control2: Point) {
  const tgt1 = control1.vecFrom(p1);
  const tgt2 = control2.vecTo(p2);

  const theta1 = tgt1.asAngle();
  const theta2 = tgt2.asAngle();

  const locusCenter = biarcLocusCenter(p1, theta1, p2, theta2);

  const positionRatio = tgt1.norm().div(tgt1.norm().add(tgt2.norm()));
  const chord = p2.vecFrom(p1);

  const positionOnChord = p1.add(chord.scale(positionRatio));

  const junctionAngle = positionOnChord.vecFrom(locusCenter).asAngle();
  const junction = locusCenter.add(
    junctionAngle.asVec().scale(locusCenter.vecFrom(p1).norm()),
  );

  return [
    bulgingSegmentUsingStartTangent(p1, junction, theta1),
    bulgingSegmentUsingEndTangent(junction, p2, theta2),
  ];
}

export function conjugateDiametersEllipse(center: Point, a: Point, b: Point) {
  const p = a.vecFrom(center);
  const pPrime = p.rotate(QUARTER_TURN);

  const d = center.add(pPrime).midPoint(b);

  const direction = a.vecFrom(d).normalize();
  const r = center.vecTo(d).norm();

  const xAxisPoint = d.add(direction.scale(r));
  const yAxisPoint = d.sub(direction.scale(r));

  const majorRadius = b.vecTo(xAxisPoint).norm();
  const minorRadius = b.vecTo(yAxisPoint).norm();

  const xAxisAngle = center.vecTo(xAxisPoint).asAngle();

  return new GenericEllipse(majorRadius, minorRadius, xAxisAngle, center);
}

export function endpointsEllipticArc(
  p1: Point,
  p2: Point,
  majorRadius: Num,
  minorRadius: Num,
  xAxisAngle: Angle,
  largeArc: Num,
  sweep: Num,
) {
  const p1Prime = p1.vecTo(p2).scale(0.5).rotate(xAxisAngle.neg());
  const p2Prime = p2.vecTo(p1).scale(0.5).rotate(xAxisAngle.neg());

  const f1 = p1Prime.x
    .square()
    .mul(minorRadius.square())
    .add(p1Prime.y.square().mul(majorRadius.square()));

  const fSign = ifTruthyElse(largeArc.equals(sweep), ONE, NEG_ONE);

  const f2 = majorRadius
    .square()
    .mul(minorRadius.square())
    .sub(f1)
    .div(f1)
    .sqrt()
    .mul(fSign);

  const cPrime = new Vec2(
    majorRadius.mul(p1Prime.y).div(minorRadius),
    minorRadius.mul(p1Prime.x).div(majorRadius).neg(),
  ).scale(f2);

  const center = p1.midPoint(p2).add(cPrime.rotate(xAxisAngle));

  const startAngle = new Vec2(
    p1Prime.x.sub(cPrime.x).div(majorRadius),
    p1Prime.y.sub(cPrime.y).div(minorRadius),
  ).asAngle();

  const endAngle = new Vec2(
    p2Prime.x.sub(cPrime.x).div(majorRadius),
    p2Prime.y.sub(cPrime.y).div(minorRadius),
  ).asAngle();

  const orientation = ifTruthyElse(sweep, NEG_ONE, ONE);

  return new EllipseArcSegment(
    majorRadius,
    minorRadius,
    startAngle,
    endAngle,
    orientation,
    center,
    xAxisAngle,
    p1,
    p2,
  );
}
