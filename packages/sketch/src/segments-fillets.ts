import { Point, Vec2, angleFromCos } from "./geom";
import { Num } from "./num";
import { BulgingSegment, LineSegment } from "./segments";
import { bulgingSegmentUsingStartTangent } from "./segments-helpers";
import { Segment } from "./types";

function lineDirection(line: LineSegment) {
  return line.p1.vecTo(line.p2).normalize();
}

export function filletLineLine(
  line1: LineSegment,
  line2: LineSegment,
  radius: Num,
) {
  const corner = line1.p2;

  const tgt1 = lineDirection(line1);
  const tgt2 = lineDirection(line2);

  const ccw = tgt1.asAngle().sub(tgt2.asAngle()).sin().sign();

  const midTangent = tgt1.add(tgt2).div(2);
  const cornerToCenter = midTangent.perp();

  const halfAngle = tgt2.asAngle().sub(cornerToCenter.asAngle());

  const distance = radius.div(halfAngle.tan());

  const newEnd1 = corner.sub(tgt1.scale(distance.mul(ccw)));
  const newStart2 = corner.add(tgt2.scale(distance.mul(ccw)));

  return [
    new LineSegment(line1.p1, newEnd1),
    bulgingSegmentUsingStartTangent(newEnd1, newStart2, tgt1.asAngle()),
    new LineSegment(newStart2, line2.p2),
  ];
}

export function projectPointOnLine(point: Point, p: Point, v: Vec2): Point {
  return p.add(v.scale(point.vecFrom(p).dot(v)));
}

function sValue(arc: BulgingSegment) {
  return arc.bulge.mul(2).div(arc.bulge.square().add(1));
}

function cValue(arc: BulgingSegment) {
  return arc.bulge.square().sub(1).neg().div(arc.bulge.square().add(1));
}

function arcTangentAtP1(arc: BulgingSegment) {
  const chord = arc.chord;
  return chord
    .scale(cValue(arc))
    .sub(chord.perp().scale(sValue(arc)))
    .normalize();
}

function arcTangentAtP2(arc: BulgingSegment) {
  const chord = arc.chord;
  return chord
    .scale(cValue(arc))
    .add(chord.perp().scale(sValue(arc)))
    .normalize();
}

export function filletLineArc(
  line: LineSegment,
  arc: BulgingSegment,
  radius: Num,
): [LineSegment, BulgingSegment, BulgingSegment] {
  const corner = line.p2;

  const tgtLine = lineDirection(line);
  const tgtArcP1 = arcTangentAtP1(arc);

  const ccw = tgtArcP1.asAngle().sub(tgtLine.asAngle()).sin().sign();

  const center = arc.center;

  const parallelP = corner.add(tgtLine.perp().scale(ccw.mul(radius)));

  const projectedCenter = projectPointOnLine(center, parallelP, tgtLine);
  const side = center.vecFrom(projectedCenter).norm();
  const hypothenuse = arc.radius.sub(ccw.mul(arc.bulge).mul(radius));

  const lastSide = hypothenuse.square().sub(side.square()).sqrt();

  const arcOrientation = arc.bulge.sign();
  const filletCenter = projectedCenter.add(
    tgtLine.scale(arcOrientation.mul(lastSide).mul(ccw)),
  );

  const lineEnd = filletCenter.sub(tgtLine.perp().scale(radius.mul(ccw)));

  const filletToArcCenter = center.vecFrom(filletCenter).normalize();
  const arcStart = filletCenter.sub(
    filletToArcCenter.scale(arcOrientation.mul(radius).mul(ccw)),
  );

  const filletArc = bulgingSegmentUsingStartTangent(
    lineEnd,
    arcStart,
    tgtLine.asAngle(),
  );
  const filletArcEndTangent = arcTangentAtP2(filletArc);
  return [
    new LineSegment(line.p1, lineEnd),
    filletArc,
    bulgingSegmentUsingStartTangent(
      arcStart,
      arc.p2,
      filletArcEndTangent.asAngle(),
    ),
  ];
}

function reverseArc(arc: BulgingSegment) {
  return new BulgingSegment(arc.p2, arc.p1, arc.bulge.neg());
}

function reverseLine(line: LineSegment) {
  return new LineSegment(line.p2, line.p1);
}

export function filletArcLine(
  arc: BulgingSegment,
  line: LineSegment,
  radius: Num,
) {
  const [line2, fillet, arc2] = filletLineArc(
    reverseLine(line),
    reverseArc(arc),
    radius,
  );
  return [reverseArc(arc2), reverseArc(fillet), reverseLine(line2)];
}

export function filletArcArc(
  arc1: BulgingSegment,
  arc2: BulgingSegment,
  radius: Num,
) {
  const arc1TgtP2 = arcTangentAtP2(arc1);
  const arc2TgtP1 = arcTangentAtP1(arc2);

  const ccw = arc2TgtP1.asAngle().sub(arc1TgtP2.asAngle()).sin().sign();
  const orientedRadius = radius.mul(ccw);
  const arc1Orientation = arc1.bulge.sign();
  const arc2Orientation = arc2.bulge.sign();

  const centersDistance = arc2.center.vecFrom(arc1.center).norm();
  const centersDirection = arc2.center.vecFrom(arc1.center).normalize();

  const side1 = arc1.radius.sub(arc1Orientation.mul(orientedRadius));
  const side2 = arc2.radius.sub(arc2Orientation.mul(orientedRadius));

  const side1Cos = side1
    .square()
    .add(centersDistance.square())
    .sub(side2.square())
    .div(centersDistance.mul(side1).mul(2));

  const angle1 = angleFromCos(side1Cos);

  const projectedFilletCenter = arc1.center.add(
    centersDirection.scale(angle1.cos().mul(side1)),
  );

  const filletCenter = projectedFilletCenter.add(
    centersDirection
      .perp()
      .scale(
        angle1
          .sin()
          .mul(side1)
          .mul(arc1Orientation)
          .mul(arc2Orientation)
          .mul(ccw),
      ),
  );

  const r1Direction = filletCenter.vecFrom(arc1.center).normalize();
  const arc1End = filletCenter.add(
    r1Direction.scale(orientedRadius.mul(arc1Orientation)),
  );

  const r2Direction = filletCenter.vecFrom(arc2.center).normalize();
  const arc2Start = filletCenter.add(
    r2Direction.scale(orientedRadius.mul(arc2Orientation)),
  );

  const startTangent = arcTangentAtP1(arc1).asAngle();
  const firstArc = bulgingSegmentUsingStartTangent(
    arc1.p1,
    arc1End,
    startTangent,
  );
  const firstArcEndTangent = arcTangentAtP2(firstArc).asAngle();
  const filletArc = bulgingSegmentUsingStartTangent(
    arc1End,
    arc2Start,
    firstArcEndTangent,
  );
  const filletArcEndTangent = arcTangentAtP2(filletArc).asAngle();
  const secondArc = bulgingSegmentUsingStartTangent(
    arc2Start,
    arc2.p2,
    filletArcEndTangent,
  );

  return [firstArc, filletArc, secondArc];
}

export function cornerFillet(
  segment1: Segment,
  segment2: Segment,
  radius: Num,
) {
  if (segment1 instanceof LineSegment && segment2 instanceof LineSegment) {
    return filletLineLine(segment1, segment2, radius);
  }

  if (segment1 instanceof LineSegment && segment2 instanceof BulgingSegment) {
    return filletLineArc(segment1, segment2, radius);
  }

  if (segment1 instanceof BulgingSegment && segment2 instanceof LineSegment) {
    return filletArcLine(segment1, segment2, radius);
  }

  if (
    segment1 instanceof BulgingSegment &&
    segment2 instanceof BulgingSegment
  ) {
    return filletArcArc(segment1, segment2, radius);
  }

  throw new Error(`Unknown segment type: ${segment1.constructor.name}`);
}
