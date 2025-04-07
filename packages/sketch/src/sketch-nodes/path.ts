import { DistanceNode, EdgeNode, PathNode, PointNode } from "./bases";

export class PathStart extends PathNode {
  public readonly nodeType = "PathStart";
  constructor(
    public readonly point: PointNode,
    public readonly cornerRadius?: DistanceNode,
  ) {
    super();
  }
}

export class PathEdge extends PathNode {
  public readonly nodeType = "PathEdge";
  constructor(
    public readonly path: PathNode,
    public readonly edge: EdgeNode,
    public readonly point: PointNode,
    public readonly cornerRadius?: DistanceNode,
  ) {
    super();
  }
}

export type AnyPathNode = PathStart | PathEdge;
