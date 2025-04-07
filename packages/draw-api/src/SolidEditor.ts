import {
  exportAsFidget,
  renderSolid,
  SolidRotationNode,
  SolidSliceNode,
} from "sketch";
import { NodeWrapper } from "./types";
import { AngleLike, asAngle, asPlane, PlaneLike } from "./convert";
import { ProfileEditor } from "./ProfileEditor";
import { AnySolidNode } from "sketch/dist/sketch-nodes/types";

export class SolidEditor implements NodeWrapper<AnySolidNode> {
  constructor(public inner: AnySolidNode) {}

  rotate(angle: AngleLike, axis: "x" | "y" | "z" = "x"): SolidEditor {
    return new SolidEditor(
      new SolidRotationNode(this.inner, asAngle(angle), axis),
    );
  }

  slice(plane: PlaneLike): ProfileEditor {
    return new ProfileEditor(new SolidSliceNode(this.inner, asPlane(plane)));
  }

  async render(
    size = 250,
    valuedVars?: Map<string, number>,
  ): Promise<Uint8ClampedArray> {
    return renderSolid(this.inner, valuedVars, size);
  }

  async fidgetExport(): Promise<string> {
    return exportAsFidget(this.inner);
  }
}
