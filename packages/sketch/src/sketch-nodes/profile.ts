import { DistanceNode, EdgeNode, PathNode, ProfileNode } from "../sketch-nodes";

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
