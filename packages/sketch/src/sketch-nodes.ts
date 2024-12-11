export class RealValueNode {}

export class DistanceNode {}

export class AngleNode {}

export class PointNode {}

export class VectorNode {}

export class EdgeNode {}

export class PathNode {}

export class ProfileNode {}

export class ConstraintNode {}

export class RealValueVariable extends RealValueNode {
  constructor(public readonly name: string) {
    super();
  }
}

export class DistanceLiteral extends DistanceNode {
  constructor(public readonly value: number) {
    super();
  }
}

export class DistanceVariable extends DistanceNode {
  constructor(public readonly name: string) {
    super();
  }
}

export class DistanceFromReal extends DistanceNode {
  constructor(public readonly value: RealValueNode) {
    super();
  }
}

export class DistanceScaled extends DistanceNode {
  constructor(
    public readonly distance: DistanceNode,
    public readonly scale: RealValueNode,
  ) {
    super();
  }
}

export class DistanceSum extends DistanceNode {
  constructor(
    public readonly left: DistanceNode,
    public readonly right: DistanceNode,
  ) {
    super();
  }
}

export class VectorNorm extends DistanceNode {
  constructor(public readonly vector: VectorNode) {
    super();
  }
}

export class ArcLength extends DistanceNode {
  constructor(
    public readonly angle: AngleNode,
    public readonly radius: DistanceNode,
  ) {
    super();
  }
}

export class AngleLiteral extends AngleNode {
  constructor(public readonly degrees: number) {
    super();
  }
}

export class AngleVariable extends AngleNode {
  constructor(public readonly name: string) {
    super();
  }
}

export class AngleSum extends AngleNode {
  constructor(
    public readonly left: AngleNode,
    public readonly right: AngleNode,
  ) {
    super();
  }
}

export class AngleDifference extends AngleNode {
  constructor(
    public readonly left: AngleNode,
    public readonly right: AngleNode,
  ) {
    super();
  }
}

export class AngleBisection extends AngleNode {
  constructor(public readonly angle: AngleNode) {
    super();
  }
}

export class VectorDirection extends AngleNode {
  constructor(public readonly vector: VectorNode) {
    super();
  }
}

export class AnglePerpendicular extends AngleNode {
  constructor(public readonly angle: AngleNode) {
    super();
  }
}

export class AngleOpposite extends AngleNode {
  constructor(public readonly angle: AngleNode) {
    super();
  }
}

export class PointVectorSum extends PointNode {
  constructor(
    public readonly point: PointNode,
    public readonly vector: VectorNode,
  ) {
    super();
  }
}

export class PointVectorDifference extends PointNode {
  constructor(
    public readonly point: PointNode,
    public readonly vector: VectorNode,
  ) {
    super();
  }
}

export class PointMidPoint extends PointNode {
  constructor(
    public readonly left: PointNode,
    public readonly right: PointNode,
  ) {
    super();
  }
}

export class PointAsVectorFromOrigin extends PointNode {
  constructor(public readonly vector: VectorNode) {
    super();
  }
}

export class Centroid extends PointNode {
  constructor(public readonly profile: ProfileNode) {
    super();
  }
}

export class VectorFromPolarCoods extends VectorNode {
  constructor(
    public readonly distance: DistanceNode,
    public readonly angle: AngleNode,
  ) {
    super();
  }
}

export class VectorFromCartesianCoords extends VectorNode {
  constructor(
    public readonly x: RealValueNode,
    public readonly y: RealValueNode,
  ) {
    super();
  }
}

export class VectorFromPoints extends VectorNode {
  constructor(
    public readonly p0: PointNode,
    public readonly p1: PointNode,
  ) {
    super();
  }
}

export class VectorFromPoint extends VectorNode {
  constructor(public readonly point: PointNode) {
    super();
  }
}

export class VectorSum extends VectorNode {
  constructor(
    public readonly left: VectorNode,
    public readonly right: VectorNode,
  ) {
    super();
  }
}

export class VectorDifference extends VectorNode {
  constructor(
    public readonly left: VectorNode,
    public readonly right: VectorNode,
  ) {
    super();
  }
}

export class VectorScaled extends VectorNode {
  constructor(
    public readonly vector: VectorNode,
    public readonly scale: RealValueNode,
  ) {
    super();
  }
}

export class VectorRotated extends VectorNode {
  constructor(
    public readonly vector: VectorNode,
    public readonly angle: AngleNode,
  ) {
    super();
  }
}

export class Line extends EdgeNode {
  constructor() {
    super();
  }
}

export class ArcFromStartControl extends EdgeNode {
  constructor(public readonly control: PointNode) {
    super();
  }
}

export class ArcFromEndControl extends EdgeNode {
  constructor(public readonly control: PointNode) {
    super();
  }
}

export class CCurve extends EdgeNode {
  constructor(public readonly control: PointNode) {
    super();
  }
}

export class SCurve extends EdgeNode {
  constructor(
    public readonly control0: PointNode,
    public readonly control1: PointNode,
  ) {
    super();
  }
}

export class PathStart extends PathNode {
  constructor(
    public readonly point: PointNode,
    public readonly cornerRadius?: DistanceNode,
  ) {
    super();
  }
}

export class PathEdge extends PathNode {
  constructor(
    public readonly path: PathNode,
    public readonly edge: EdgeNode,
    public readonly point: PointNode,
    public readonly cornerRadius?: DistanceNode,
  ) {
    super();
  }
}

export class PathClose extends ProfileNode {
  constructor(
    public readonly path: PathNode,
    public readonly edge: EdgeNode,
  ) {
    super();
  }
}

export class PathOpenEnd extends ProfileNode {
  constructor(public readonly path: PathNode) {
    super();
  }
}

export class Circle extends ProfileNode {
  constructor(public readonly radius: DistanceNode) {
    super();
  }
}

export class Box extends ProfileNode {
  constructor(
    public readonly width: DistanceNode,
    public readonly height: DistanceNode,
  ) {
    super();
  }
}

export class Translation extends ProfileNode {
  constructor(
    public readonly profile: ProfileNode,
    public readonly vector: VectorNode,
  ) {
    super();
  }
}

export class Rotation extends ProfileNode {
  constructor(
    public readonly profile: ProfileNode,
    public readonly angle: AngleNode,
  ) {
    super();
  }
}

export class Scale extends ProfileNode {
  constructor(
    public readonly profile: ProfileNode,
    public readonly factor: RealValueNode,
  ) {
    super();
  }
}

export class Union extends ProfileNode {
  constructor(
    public readonly left: ProfileNode,
    public readonly right: ProfileNode,
  ) {
    super();
  }
}

export class SmoothUnion extends ProfileNode {
  constructor(
    public readonly left: ProfileNode,
    public readonly right: ProfileNode,
    public readonly radius: DistanceNode,
  ) {
    super();
  }
}

export class Intersection extends ProfileNode {
  constructor(
    public readonly left: ProfileNode,
    public readonly right: ProfileNode,
  ) {
    super();
  }
}

export class SmoothIntersection extends ProfileNode {
  constructor(
    public readonly left: ProfileNode,
    public readonly right: ProfileNode,
    public readonly radius: DistanceNode,
  ) {
    super();
  }
}

export class Difference extends ProfileNode {
  constructor(
    public readonly left: ProfileNode,
    public readonly right: ProfileNode,
  ) {
    super();
  }
}

export class SmoothDifference extends ProfileNode {
  constructor(
    public readonly left: ProfileNode,
    public readonly right: ProfileNode,
    public readonly radius: DistanceNode,
  ) {
    super();
  }
}

export class Shell extends ProfileNode {
  constructor(
    public readonly profile: ProfileNode,
    public readonly thickness: DistanceNode,
  ) {
    super();
  }
}

export class Morph extends ProfileNode {
  constructor(
    public readonly start: ProfileNode,
    public readonly end: ProfileNode,
    public readonly t: DistanceNode,
  ) {
    super();
  }
}

export class Dilate extends ProfileNode {
  constructor(
    public readonly profile: ProfileNode,
    public readonly factor: DistanceNode,
  ) {
    super();
  }
}

export class SignedDistanceToProfile extends RealValueNode {
  constructor(
    public readonly profile: ProfileNode,
    public readonly point: PointNode,
  ) {
    super();
  }
}

export class ConstraintOnDistance extends ConstraintNode {
  constructor(
    public readonly distance: DistanceNode,
    public readonly target: DistanceNode,
    public readonly weigth: DistanceNode | undefined,
  ) {
    super();
  }
}

export class ConstraintOnAngle extends ConstraintNode {
  constructor(
    public readonly angle: AngleNode,
    public readonly target: AngleNode,
    public readonly weigth: DistanceNode | undefined,
  ) {
    super();
  }
}

export class ConstraintOnPoint extends ConstraintNode {
  constructor(
    public readonly point: PointNode,
    public readonly target: PointNode,
    public readonly weigth: DistanceNode | undefined,
  ) {
    super();
  }
}

export class ConstraintOnProfileBoundary extends ConstraintNode {
  constructor(
    public readonly profile: ProfileNode,
    public readonly point: PointNode,
    public readonly signedDistance: RealValueNode | undefined,
    public readonly weigth: DistanceNode | undefined,
  ) {
    super();
  }
}
