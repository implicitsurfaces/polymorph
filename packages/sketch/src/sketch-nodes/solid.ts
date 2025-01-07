import { DistanceNode, ProfileNode, SolidNode } from "../sketch-nodes";

export class SphereNode extends SolidNode {
  constructor(public readonly radius: DistanceNode) {
    super();
  }
}

export class ConeNode extends SolidNode {
  constructor(
    public readonly radius: DistanceNode,
    public readonly height: DistanceNode,
  ) {
    super();
  }
}

export class ExtrusionNode extends SolidNode {
  constructor(
    public readonly profile: ProfileNode,
    public readonly height: DistanceNode,
  ) {
    super();
  }
}
