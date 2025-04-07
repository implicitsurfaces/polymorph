import { AngleNode, DistanceNode, EdgeNode, PointNode } from "./bases";

export class Line extends EdgeNode {
  public readonly nodeType = "Line";
  constructor() {
    super();
  }
}

export class ArcFromStartControl extends EdgeNode {
  public readonly nodeType = "ArcFromStartControl";
  constructor(public readonly control: PointNode) {
    super();
  }
}

export class ArcFromEndControl extends EdgeNode {
  public readonly nodeType = "ArcFromEndControl";
  constructor(public readonly control: PointNode) {
    super();
  }
}

export class CCurve extends EdgeNode {
  public readonly nodeType = "CCurve";
  constructor(public readonly control: PointNode) {
    super();
  }
}

export class SCurve extends EdgeNode {
  public readonly nodeType = "SCurve";
  constructor(
    public readonly control0: PointNode,
    public readonly control1: PointNode,
  ) {
    super();
  }
}

export class EllipseArcNode extends EdgeNode {
  public readonly nodeType = "EllipseArc";
  constructor(
    public readonly majorRadius: DistanceNode,
    public readonly minorRadius: DistanceNode,
    public readonly rotation: AngleNode,
    public readonly largeArc = false,
    public readonly sweep = false,
  ) {
    super();
  }
}

export type AnyEdgeNode =
  | Line
  | ArcFromStartControl
  | ArcFromEndControl
  | CCurve
  | SCurve
  | EllipseArcNode;
