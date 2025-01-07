import {
  AngleNode,
  DistanceNode,
  PointNode,
  ProfileNode,
  RealValueNode,
  VectorNode,
} from "./bases";

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

export class GradientAt extends VectorNode {
  constructor(
    public readonly field: ProfileNode,
    public readonly point: PointNode,
  ) {
    super();
  }
}
