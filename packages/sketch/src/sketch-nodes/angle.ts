import { AngleNode, VectorNode } from "../sketch-nodes";

export class AngleLiteral extends AngleNode {
  public readonly nodeType = "AngleLiteral";
  constructor(public readonly degrees: number) {
    super();
  }
}

export class AngleVariable extends AngleNode {
  public readonly nodeType = "AngleVariable";
  constructor(public readonly name: string) {
    super();
  }
}

export class AngleSum extends AngleNode {
  public readonly nodeType = "AngleSum";
  constructor(
    public readonly left: AngleNode,
    public readonly right: AngleNode,
  ) {
    super();
  }
}

export class AngleDifference extends AngleNode {
  public readonly nodeType = "AngleDifference";
  constructor(
    public readonly left: AngleNode,
    public readonly right: AngleNode,
  ) {
    super();
  }
}

export class AngleBisection extends AngleNode {
  public readonly nodeType = "AngleBisection";
  constructor(public readonly angle: AngleNode) {
    super();
  }
}

export class VectorDirection extends AngleNode {
  public readonly nodeType = "VectorDirection";
  constructor(public readonly vector: VectorNode) {
    super();
  }
}

export class AnglePerpendicular extends AngleNode {
  public readonly nodeType = "AnglePerpendicular";
  constructor(public readonly angle: AngleNode) {
    super();
  }
}

export class AngleOpposite extends AngleNode {
  public readonly nodeType = "AngleOpposite";
  constructor(public readonly angle: AngleNode) {
    super();
  }
}

export type AnyAngleNode =
  | AngleLiteral
  | AngleVariable
  | AngleSum
  | AngleDifference
  | AngleBisection
  | VectorDirection
  | AnglePerpendicular
  | AngleOpposite;
