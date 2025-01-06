import { renderSolid, SolidNode, SolidRotationNode } from "sketch";
import { NodeWrapper } from "./types";
import { AngleLike, asAngle } from "./convert";

export class SolidEditor implements NodeWrapper<SolidNode> {
  constructor(public inner: SolidNode) {}

  rotate(angle: AngleLike, axis: "x" | "y" | "z" = "x"): SolidEditor {
    return new SolidEditor(
      new SolidRotationNode(this.inner, asAngle(angle), axis),
    );
  }

  async render(
    size = 250,
    valuedVars?: Map<string, number>,
  ): Promise<Uint8ClampedArray> {
    return renderSolid(this.inner, valuedVars, size);
  }
}
