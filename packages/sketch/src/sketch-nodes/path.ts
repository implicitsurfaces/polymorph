import { DistanceNode, EdgeNode, PathNode, PointNode } from "../sketch-nodes";

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
