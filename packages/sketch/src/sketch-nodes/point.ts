import { PointNode, VectorNode } from "./bases";

export class PointVectorSum extends PointNode {
  public readonly nodeType = "PointVectorSum";
  constructor(
    public readonly point: PointNode,
    public readonly vector: VectorNode,
  ) {
    super();
  }
}

export class PointVectorDifference extends PointNode {
  public readonly nodeType = "PointVectorDifference";
  constructor(
    public readonly point: PointNode,
    public readonly vector: VectorNode,
  ) {
    super();
  }
}

export class PointMidPoint extends PointNode {
  public readonly nodeType = "PointMidPoint";
  constructor(
    public readonly left: PointNode,
    public readonly right: PointNode,
  ) {
    super();
  }
}

export class PointAsVectorFromOrigin extends PointNode {
  public readonly nodeType = "PointAsVectorFromOrigin";
  constructor(public readonly vector: VectorNode) {
    super();
  }
}

export type AnyPointNode =
  | PointVectorSum
  | PointVectorDifference
  | PointMidPoint
  | PointAsVectorFromOrigin;
