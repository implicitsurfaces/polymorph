import { DistanceNode, ProfileNode, SolidNode } from "./bases";

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

export class ConeSurfaceNode extends SolidNode {
  constructor(
    public readonly radius: DistanceNode,
    public readonly height: DistanceNode,
  ) {
    super();
  }
}

export class EllipticConeNode extends SolidNode {
  constructor(
    public readonly majorRadius: DistanceNode,
    public readonly minorRadius: DistanceNode,
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
