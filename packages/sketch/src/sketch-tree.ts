import { Num, asNum } from "./num";
import { Vec2, Angle, angleFromDeg, Point } from "./geom";
import {
  DistanceNode,
  DistanceScaled,
  DistanceSum,
  RealValueNode,
  VectorNorm,
  VectorNode,
  AngleNode,
  ArcLength,
  VectorFromPolarCoods,
  AngleLiteral,
  AngleSum,
  AngleDifference,
  AnglePerpendicular,
  AngleOpposite,
  AngleBisection,
  VectorDirection,
  PointNode,
  PointAsVectorFromOrigin,
  PointVectorSum,
  PointVectorDifference,
  PointMidPoint,
  VectorFromPoint,
  VectorFromPoints,
  VectorFromCartesianCoords,
  VectorSum,
  VectorDifference,
  VectorScaled,
  EdgeNode,
  Line,
  ArcFromStartControl,
  ArcFromEndControl,
  BiarcC,
  BiarcS,
  PathStart,
  PathEdge,
  PathNode,
  PathOpenEnd,
  PathClose,
  Circle as CircleNode,
  Box as BoxNode,
  Translation as TranslationNode,
  Rotation as RotationNode,
  Scale as ScaleNode,
  Union as UnionNode,
  SmoothUnion as SmoothUnionNode,
  Difference as DifferenceNode,
  SmoothDifference as SmoothDifferenceNode,
  Intersection as IntersectionNode,
  SmoothIntersection as SmoothIntersectionNode,
  Shell as ShellNode,
  Morph as MorphNode,
  SignedDistanceToProfile,
  DistanceLiteral,
  Dilate,
} from "./sketch-nodes";
import { LineSegment } from "./segments";
import {
  biarcC,
  biarcS,
  bulgingSegmentUsingEndControl,
  bulgingSegmentUsingStartControl,
} from "./segments-helpers";
import { DistField, Segment } from "./types";
import { Circle, ClosedPath, OpenPath, Box } from "./profiles";
import {
  Rotation,
  Translation,
  Scaling,
  Union,
  SmoothUnion,
  Difference,
  SmoothDifference,
  Intersection,
  SmoothIntersection,
  Morph,
  Shell,
  Dilatation,
} from "./sdf-operations";
import { cornerFillet } from "./segments-fillets";
import { memoizeNodeEval } from "./utils/cache";

export function evalRealValue(value: RealValueNode): Num {
  if (value instanceof DistanceNode) {
    return evalDistance(value);
  }
  if (typeof value === "number") {
    return asNum(value);
  }
  if (value instanceof SignedDistanceToProfile) {
    return evalProfile(value.profile).distanceTo(evalPoint(value.point));
  }

  throw new Error(`Unknown real value: ${value}`);
}

export const evalDistance = memoizeNodeEval(function (
  distance: DistanceNode,
): Num {
  if (distance instanceof DistanceLiteral) {
    return asNum(distance.value);
  }
  if (distance instanceof DistanceScaled) {
    return evalDistance(distance.distance).mul(evalRealValue(distance.scale));
  }
  if (distance instanceof DistanceSum) {
    return evalDistance(distance.left).add(evalDistance(distance.right));
  }
  if (distance instanceof VectorNorm) {
    return evalVector(distance.vector).norm();
  }
  if (distance instanceof ArcLength) {
    return evalAngle(distance.angle).asRad().mul(evalDistance(distance.radius));
  }

  throw new Error(`Unknown distance: ${distance.constructor.name}`);
});

export const evalAngle = memoizeNodeEval(function (angle: AngleNode): Angle {
  if (angle instanceof AngleLiteral) {
    return angleFromDeg(evalRealValue(angle.degrees));
  }

  if (angle instanceof AngleSum) {
    return evalAngle(angle.left).add(evalAngle(angle.right));
  }

  if (angle instanceof AngleDifference) {
    return evalAngle(angle.left).sub(evalAngle(angle.right));
  }

  if (angle instanceof AnglePerpendicular) {
    return evalAngle(angle.angle).perp();
  }

  if (angle instanceof AngleOpposite) {
    return evalAngle(angle.angle).opposite();
  }

  if (angle instanceof AngleBisection) {
    return evalAngle(angle.angle).half();
  }

  if (angle instanceof VectorDirection) {
    return evalVector(angle.vector).asAngle();
  }

  throw new Error(`Unknown angle: ${angle.constructor.name}`);
});

export const evalPoint = memoizeNodeEval(function (point: PointNode): Point {
  if (point instanceof PointAsVectorFromOrigin) {
    return evalVector(point.vector).pointFromOrigin();
  }

  if (point instanceof PointVectorSum) {
    return evalPoint(point.point).add(evalVector(point.vector));
  }

  if (point instanceof PointVectorDifference) {
    return evalPoint(point.point).sub(evalVector(point.vector));
  }

  if (point instanceof PointMidPoint) {
    return evalPoint(point.left).midPoint(evalPoint(point.right));
  }

  throw new Error(`Unknown point: ${point.constructor.name}`);
});

export const evalVector = memoizeNodeEval(function evalVector(
  vector: VectorNode,
): Vec2 {
  if (vector instanceof VectorFromPolarCoods) {
    return evalAngle(vector.angle).asVec().scale(evalDistance(vector.distance));
  }

  if (vector instanceof VectorFromPoint) {
    return evalPoint(vector.point).vecFromOrigin();
  }

  if (vector instanceof VectorFromPoints) {
    return evalPoint(vector.p0).vecTo(evalPoint(vector.p1));
  }

  if (vector instanceof VectorFromCartesianCoords) {
    return new Vec2(evalRealValue(vector.x), evalRealValue(vector.y));
  }

  if (vector instanceof VectorSum) {
    return evalVector(vector.left).add(evalVector(vector.right));
  }

  if (vector instanceof VectorDifference) {
    return evalVector(vector.left).sub(evalVector(vector.right));
  }

  if (vector instanceof VectorScaled) {
    return evalVector(vector.vector).scale(evalRealValue(vector.scale));
  }

  throw new Error(`Unknown vector: ${vector.constructor.name}`);
});

export const evalEdge = memoizeNodeEval(function (
  edge: EdgeNode,
): (p0: Point, p1: Point) => Segment[] {
  if (edge instanceof Line) {
    return (p0: Point, p1: Point) => [new LineSegment(p0, p1)];
  }

  if (edge instanceof ArcFromStartControl) {
    return (p0: Point, p1: Point) => [
      bulgingSegmentUsingStartControl(p0, p1, evalPoint(edge.control)),
    ];
  }

  if (edge instanceof ArcFromEndControl) {
    return (p0: Point, p1: Point) => [
      bulgingSegmentUsingEndControl(p0, p1, evalPoint(edge.control)),
    ];
  }

  if (edge instanceof BiarcC) {
    return (p0: Point, p1: Point) => biarcC(p0, p1, evalPoint(edge.control));
  }

  if (edge instanceof BiarcS) {
    return (p0: Point, p1: Point) =>
      biarcS(p0, p1, evalPoint(edge.control0), evalPoint(edge.control1));
  }

  throw new Error(`Unknown edge: ${edge.constructor.name}`);
});

class PartialPath {
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

export const evalPath = memoizeNodeEval(function (node: PathNode): PartialPath {
  if (node instanceof PathStart) {
    const startPoint = evalPoint(node.point);
    const startRadius =
      node.cornerRadius || node.cornerRadius === 0
        ? evalDistance(node.cornerRadius)
        : undefined;

    return new PartialPath(startPoint, startRadius);
  }

  if (node instanceof PathEdge) {
    const path = evalPath(node.path);
    const point = evalPoint(node.point);
    const radius =
      node.cornerRadius || node.cornerRadius === 0
        ? evalDistance(node.cornerRadius)
        : undefined;

    const edgeFn = evalEdge(node.edge);

    return path.append(edgeFn, point, radius);
  }

  throw new Error(`Unknown path: ${node.constructor.name}`);
});

export const evalProfile = memoizeNodeEval(function (
  node: PathNode,
): DistField {
  if (node instanceof PathClose) {
    const path = evalPath(node.path);
    const edgeFn = evalEdge(node.edge);

    const segments = path.close(edgeFn);
    return new ClosedPath(segments);
  }

  if (node instanceof PathOpenEnd) {
    const path = evalPath(node.path);
    return new OpenPath(path.segments);
  }

  if (node instanceof CircleNode) {
    const radius = evalDistance(node.radius);
    return new Circle(radius);
  }

  if (node instanceof BoxNode) {
    const width = evalDistance(node.width);
    const height = evalDistance(node.height);
    return new Box(width, height);
  }

  if (node instanceof TranslationNode) {
    return new Translation(evalVector(node.vector), evalProfile(node.profile));
  }

  if (node instanceof RotationNode) {
    return new Rotation(evalAngle(node.angle), evalProfile(node.profile));
  }

  if (node instanceof ScaleNode) {
    return new Scaling(evalRealValue(node.factor), evalProfile(node.profile));
  }

  if (node instanceof UnionNode) {
    return new Union(evalProfile(node.left), evalProfile(node.right));
  }

  if (node instanceof SmoothUnionNode) {
    return new SmoothUnion(
      evalDistance(node.radius),
      evalProfile(node.left),
      evalProfile(node.right),
    );
  }

  if (node instanceof DifferenceNode) {
    return new Difference(evalProfile(node.left), evalProfile(node.right));
  }

  if (node instanceof SmoothDifferenceNode) {
    return new SmoothDifference(
      evalDistance(node.radius),
      evalProfile(node.left),
      evalProfile(node.right),
    );
  }

  if (node instanceof IntersectionNode) {
    return new Intersection(evalProfile(node.left), evalProfile(node.right));
  }

  if (node instanceof SmoothIntersectionNode) {
    return new SmoothIntersection(
      evalDistance(node.radius),
      evalProfile(node.left),
      evalProfile(node.right),
    );
  }

  if (node instanceof ShellNode) {
    return new Shell(evalDistance(node.thickness), evalProfile(node.profile));
  }

  if (node instanceof MorphNode) {
    return new Morph(
      evalRealValue(node.t),
      evalProfile(node.start),
      evalProfile(node.end),
    );
  }

  if (node instanceof Dilate) {
    return new Dilatation(
      evalRealValue(node.factor),
      evalProfile(node.profile),
    );
  }

  throw new Error(`Unknown profile: ${node.constructor.name}`);
});
