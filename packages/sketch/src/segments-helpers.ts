import { Angle, Point, Vec2 } from "./geom";
import { BulgingSegment } from "./segments";

export function bulgingSegmentUsingStartTangent(
  p1: Point,
  p2: Point,
  startTangent: Angle,
) {
  const chord = p1.vecTo(p2);

  const tgt = startTangent.asVec();

  const s = chord.cross(tgt);
  const c = chord.dot(tgt);

  const bulge = s.neg().div(c.add(c.square().add(s.square()).sqrt()));
  return new BulgingSegment(p1, p2, bulge);
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

  const bulge = s.neg().div(c.add(c.square().add(s.square()).sqrt()));
  return new BulgingSegment(p1, p2, bulge);
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
