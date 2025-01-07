import {
  AngleNode,
  DistanceNode,
  RealValueNode,
  VectorNode,
} from "../sketch-nodes";

export class DistanceLiteral extends DistanceNode {
  constructor(public readonly value: number) {
    super();
  }
}

export class DistanceVariable extends DistanceNode {
  constructor(
    public readonly name: string,
    public readonly min?: number,
    public readonly max?: number,
  ) {
    super();
  }
}

export class DistanceFromReal extends DistanceNode {
  constructor(public readonly value: RealValueNode) {
    super();
  }
}

export class DistanceScaled extends DistanceNode {
  constructor(
    public readonly distance: DistanceNode,
    public readonly scale: RealValueNode,
  ) {
    super();
  }
}

export class DistanceSum extends DistanceNode {
  constructor(
    public readonly left: DistanceNode,
    public readonly right: DistanceNode,
  ) {
    super();
  }
}

export class VectorNorm extends DistanceNode {
  constructor(public readonly vector: VectorNode) {
    super();
  }
}

export class ArcLength extends DistanceNode {
  constructor(
    public readonly angle: AngleNode,
    public readonly radius: DistanceNode,
  ) {
    super();
  }
}
