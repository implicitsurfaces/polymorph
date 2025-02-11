import { AngleNode, DistanceNode, EdgeNode, PointNode } from "./bases";

export class Line extends EdgeNode {
  constructor() {
    super();
  }
}

export class ArcFromStartControl extends EdgeNode {
  constructor(public readonly control: PointNode) {
    super();
  }
}

export class ArcFromEndControl extends EdgeNode {
  constructor(public readonly control: PointNode) {
    super();
  }
}

export class CCurve extends EdgeNode {
  constructor(public readonly control: PointNode) {
    super();
  }
}

export class SCurve extends EdgeNode {
  constructor(
    public readonly control0: PointNode,
    public readonly control1: PointNode,
  ) {
    super();
  }
}

export class EllipseArcNode extends EdgeNode {
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
