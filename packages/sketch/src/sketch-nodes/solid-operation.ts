import { AngleNode, SolidNode } from "../sketch-nodes";

export class SolidRotationNode extends SolidNode {
  constructor(
    public readonly solid: SolidNode,
    public readonly angle: AngleNode,
    public readonly axis: "x" | "y" | "z",
  ) {
    super();
  }
}
