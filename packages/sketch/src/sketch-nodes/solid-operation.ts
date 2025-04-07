import { AngleNode, SolidNode } from "./bases";

export class SolidRotationNode extends SolidNode {
  public readonly nodeType = "SolidRotation";
  constructor(
    public readonly solid: SolidNode,
    public readonly angle: AngleNode,
    public readonly axis: "x" | "y" | "z",
  ) {
    super();
  }
}

export type AnySolidOperationNode = SolidRotationNode;
