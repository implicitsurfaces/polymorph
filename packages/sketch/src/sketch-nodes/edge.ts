import { EdgeNode, PointNode } from "../sketch-nodes";

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
