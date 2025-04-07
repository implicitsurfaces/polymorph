import {
  AngleNode,
  ConstraintNode,
  DistanceNode,
  PointNode,
  ProfileNode,
  RealValueOrNumber,
} from "./bases";

export class ConstraintOnDistance extends ConstraintNode {
  public readonly nodeType = "ConstraintOnDistance";
  constructor(
    public readonly distance: DistanceNode,
    public readonly target: DistanceNode,
    public readonly weigth: DistanceNode | undefined,
  ) {
    super();
  }
}

export class ConstraintOnAngle extends ConstraintNode {
  public readonly nodeType = "ConstraintOnAngle";
  constructor(
    public readonly angle: AngleNode,
    public readonly target: AngleNode,
    public readonly weigth: DistanceNode | undefined,
  ) {
    super();
  }
}

export class ConstraintOnPoint extends ConstraintNode {
  public readonly nodeType = "ConstraintOnPoint";
  constructor(
    public readonly point: PointNode,
    public readonly target: PointNode,
    public readonly weigth: DistanceNode | undefined,
  ) {
    super();
  }
}

export class ConstraintOnProfileBoundary extends ConstraintNode {
  public readonly nodeType = "ConstraintOnProfileBoundary";
  constructor(
    public readonly profile: ProfileNode,
    public readonly point: PointNode,
    public readonly signedDistance: RealValueOrNumber | undefined,
    public readonly weigth: DistanceNode | undefined,
  ) {
    super();
  }
}

export type AnyConstraintNode =
  | ConstraintOnDistance
  | ConstraintOnAngle
  | ConstraintOnPoint
  | ConstraintOnProfileBoundary;
