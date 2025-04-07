import { Point } from "../../../geom";
import { Num } from "../../../num";
import { cornerFillet } from "../../../segments-fillets";
import { Segment } from "../../../types";

export class PartialPath {
  public segments: Segment[];
  private endPoint: Point;

  private firstRadius?: Num;
  private endRadius?: Num;
  private first: Point;

  constructor(point: Point, radius?: Num) {
    this.segments = [];
    this.endPoint = point;
    this.firstRadius = radius;
    this.first = point;
  }

  _appendSegments(segments: Segment[]): PartialPath {
    if (this.endRadius) {
      const filletedCorner = cornerFillet(
        this.segments.pop()!,
        segments.shift()!,
        this.endRadius,
      );
      this.segments.push(...filletedCorner);
    }
    this.segments.push(...segments);
    return this;
  }

  append(
    segmentFn: (p0: Point, p1: Point) => Segment[],
    point: Point,
    radius?: Num,
  ): PartialPath {
    const segments = segmentFn(this.endPoint, point);
    this._appendSegments(segments);
    this.endPoint = point;
    this.endRadius = radius;
    return this;
  }

  close(segmentFn: (p0: Point, p1: Point) => Segment[]): Segment[] {
    const segments = segmentFn(this.endPoint, this.first);
    this._appendSegments(segments);

    if (this.firstRadius) {
      const filletedCorner = cornerFillet(
        this.segments.pop()!,
        this.segments.shift()!,
        this.firstRadius,
      );
      this.segments.push(...filletedCorner);
    }
    return this.segments;
  }
}
