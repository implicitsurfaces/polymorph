import { AngleNode, VectorNode } from "../sketch-nodes";

export class AngleLiteral extends AngleNode {
  constructor(public readonly degrees: number) {
    super();
  }
}

export class AngleVariable extends AngleNode {
  constructor(public readonly name: string) {
    super();
  }
}

export class AngleSum extends AngleNode {
  constructor(
    public readonly left: AngleNode,
    public readonly right: AngleNode,
  ) {
    super();
  }
}

export class AngleDifference extends AngleNode {
  constructor(
    public readonly left: AngleNode,
    public readonly right: AngleNode,
  ) {
    super();
  }
}

export class AngleBisection extends AngleNode {
  constructor(public readonly angle: AngleNode) {
    super();
  }
}

export class VectorDirection extends AngleNode {
  constructor(public readonly vector: VectorNode) {
    super();
  }
}

export class AnglePerpendicular extends AngleNode {
  constructor(public readonly angle: AngleNode) {
    super();
  }
}

export class AngleOpposite extends AngleNode {
  constructor(public readonly angle: AngleNode) {
    super();
  }
}
