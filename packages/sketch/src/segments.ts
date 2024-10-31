import { Point, SolidAngle, Vec2, two_vectors_angle } from "./geom";
import { Num } from "./num";
import { clamp } from "./num-ops";
import { Segment } from "./types";

export class LineSegment implements Segment {
  private segment: Vec2;
  constructor(
    readonly p1: Point,
    readonly p2: Point,
  ) {
    this.segment = p1.vec_to(p2);
  }

  solidAngle(p: Point): SolidAngle {
    const a = this.p1.vec_to(p);
    const b = this.p2.vec_to(p);

    return new SolidAngle(0).add_angle(two_vectors_angle(a, b));
  }

  distanceTo(p: Point): Num {
    const startToP = p.vec_from(this.p1);
    const parametricPosition = startToP
      .dot(this.segment)
      .div(this.segment.dot(this.segment));

    const clampedPosition = clamp(parametricPosition, 0, 1);

    const projectedPoint = this.p1.add(this.segment.scale(clampedPosition));
    return p.vec_from(projectedPoint).norm();
  }
}
