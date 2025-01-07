import {
  AngleNode,
  ConstraintNode,
  DistanceNode,
  PointNode,
  ProfileNode,
  RealValueNode,
} from "../sketch-nodes";

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
