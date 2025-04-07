import {
  AngleNode,
  DistanceNode,
  PointNode,
  ProfileNode,
  RealValueOrNumber,
  VectorNode,
} from "./bases";

export class VectorFromPolarCoods extends VectorNode {
  public readonly nodeType = "VectorFromPolarCoods";
  constructor(
    public readonly distance: DistanceNode,
    public readonly angle: AngleNode,
  ) {
    super();
  }
}

export class VectorFromCartesianCoords extends VectorNode {
  public readonly nodeType = "VectorFromCartesianCoords";
  constructor(
    public readonly x: RealValueOrNumber,
    public readonly y: RealValueOrNumber,
  ) {
    super();
  }
}

export class VectorFromPoints extends VectorNode {
  public readonly nodeType = "VectorFromPoints";
  constructor(
    public readonly p0: PointNode,
    public readonly p1: PointNode,
  ) {
    super();
  }
}

export class VectorFromPoint extends VectorNode {
  public readonly nodeType = "VectorFromPoint";
  constructor(public readonly point: PointNode) {
    super();
  }
}

export class VectorSum extends VectorNode {
  public readonly nodeType = "VectorSum";
  constructor(
    public readonly left: VectorNode,
    public readonly right: VectorNode,
  ) {
    super();
  }
}

export class VectorDifference extends VectorNode {
  public readonly nodeType = "VectorDifference";
  constructor(
    public readonly left: VectorNode,
    public readonly right: VectorNode,
  ) {
    super();
  }
}

export class VectorScaled extends VectorNode {
  public readonly nodeType = "VectorScaled";
  constructor(
    public readonly vector: VectorNode,
    public readonly scale: RealValueOrNumber,
  ) {
    super();
  }
}

export class VectorRotated extends VectorNode {
  public readonly nodeType = "VectorRotated";
  constructor(
    public readonly vector: VectorNode,
    public readonly angle: AngleNode,
  ) {
    super();
  }
}

export class GradientAt extends VectorNode {
  public readonly nodeType = "GradientAt";
  constructor(
    public readonly field: ProfileNode,
    public readonly point: PointNode,
  ) {
    super();
  }
}

export type AnyVectorNode =
  | VectorFromPolarCoods
  | VectorFromCartesianCoords
  | VectorFromPoints
  | VectorFromPoint
  | VectorSum
  | VectorDifference
  | VectorScaled
  | VectorRotated
  | GradientAt;
