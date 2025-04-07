import {
  AngleNode,
  DistanceNode,
  RealValueOrNumber,
  Vector3Node,
  VectorNode,
} from "./bases";

export class DistanceLiteral extends DistanceNode {
  public readonly nodeType = "DistanceLiteral";
  constructor(public readonly value: number) {
    super();
  }
}

export class DistanceVariable extends DistanceNode {
  public readonly nodeType = "DistanceVariable";
  constructor(
    public readonly name: string,
    public readonly min?: number,
    public readonly max?: number,
  ) {
    super();
  }
}

export class DistanceFromReal extends DistanceNode {
  public readonly nodeType = "DistanceFromReal";
  constructor(public readonly value: RealValueOrNumber) {
    super();
  }
}

export class DistanceScaled extends DistanceNode {
  public readonly nodeType = "DistanceScaled";
  constructor(
    public readonly distance: DistanceNode,
    public readonly scale: RealValueOrNumber,
  ) {
    super();
  }
}

export class DistanceSum extends DistanceNode {
  public readonly nodeType = "DistanceSum";
  constructor(
    public readonly left: DistanceNode,
    public readonly right: DistanceNode,
  ) {
    super();
  }
}

export class VectorNorm extends DistanceNode {
  public readonly nodeType = "VectorNorm";
  constructor(public readonly vector: VectorNode) {
    super();
  }
}

export class Vector3Norm extends DistanceNode {
  public readonly nodeType = "Vector3Norm";
  constructor(public readonly vector: Vector3Node) {
    super();
  }
}

export class ArcLength extends DistanceNode {
  public readonly nodeType = "ArcLength";
  constructor(
    public readonly angle: AngleNode,
    public readonly radius: DistanceNode,
  ) {
    super();
  }
}

export type AnyDistanceNode =
  | DistanceLiteral
  | DistanceVariable
  | DistanceFromReal
  | DistanceScaled
  | DistanceSum
  | VectorNorm
  | Vector3Norm
  | ArcLength;
