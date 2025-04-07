import {
  DistanceNode,
  EdgeNode,
  PathNode,
  PlaneNode,
  ProfileNode,
  SolidNode,
} from "./bases";

export class PathClose extends ProfileNode {
  public readonly nodeType = "PathClose";
  constructor(
    public readonly path: PathNode,
    public readonly edge: EdgeNode,
  ) {
    super();
  }
}

export class PathOpenEnd extends ProfileNode {
  public readonly nodeType = "PathOpenEnd";
  constructor(public readonly path: PathNode) {
    super();
  }
}

export class Circle extends ProfileNode {
  public readonly nodeType = "Circle";
  constructor(public readonly radius: DistanceNode) {
    super();
  }
}

export class Box extends ProfileNode {
  public readonly nodeType = "Box";
  constructor(
    public readonly width: DistanceNode,
    public readonly height: DistanceNode,
  ) {
    super();
  }
}

export class SolidSliceNode extends ProfileNode {
  public readonly nodeType = "SolidSlice";
  constructor(
    public readonly solid: SolidNode,
    public readonly plane: PlaneNode,
  ) {
    super();
  }
}

export class EllipseNode extends ProfileNode {
  public readonly nodeType = "Ellipse";
  constructor(
    public readonly majorRadius: DistanceNode,
    public readonly minorRadius: DistanceNode,
  ) {
    super();
  }
}

export type AnyBaseProfileNode =
  | PathClose
  | PathOpenEnd
  | Circle
  | Box
  | SolidSliceNode
  | EllipseNode;
