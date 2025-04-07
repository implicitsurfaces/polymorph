import { DistanceNode, ProfileNode, SolidNode } from "./bases";

export class SphereNode extends SolidNode {
  public readonly nodeType = "Sphere";
  constructor(public readonly radius: DistanceNode) {
    super();
  }
}

export class ConeNode extends SolidNode {
  public readonly nodeType = "Cone";
  constructor(
    public readonly radius: DistanceNode,
    public readonly height: DistanceNode,
  ) {
    super();
  }
}

export class ConeSurfaceNode extends SolidNode {
  public readonly nodeType = "ConeSurface";
  constructor(
    public readonly radius: DistanceNode,
    public readonly height: DistanceNode,
  ) {
    super();
  }
}

export class ExtrusionNode extends SolidNode {
  public readonly nodeType = "Extrusion";
  constructor(
    public readonly profile: ProfileNode,
    public readonly height: DistanceNode,
  ) {
    super();
  }
}

export type AnyBasicSolidNode =
  | SphereNode
  | ConeNode
  | ConeSurfaceNode
  | ExtrusionNode;
