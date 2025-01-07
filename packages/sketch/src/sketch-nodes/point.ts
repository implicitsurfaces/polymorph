import { PointNode, ProfileNode, VectorNode } from "../sketch-nodes";

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
