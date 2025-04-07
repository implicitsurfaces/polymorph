import {
  AngleNode,
  DistanceNode,
  ProfileNode,
  RealValueOrNumber,
  VectorNode,
} from "./bases";

export class Translation extends ProfileNode {
  public readonly nodeType = "Translation";
  constructor(
    public readonly profile: ProfileNode,
    public readonly vector: VectorNode,
  ) {
    super();
  }
}

export class Rotation extends ProfileNode {
  public readonly nodeType = "Rotation";
  constructor(
    public readonly profile: ProfileNode,
    public readonly angle: AngleNode,
  ) {
    super();
  }
}

export class Scale extends ProfileNode {
  public readonly nodeType = "Scale";
  constructor(
    public readonly profile: ProfileNode,
    public readonly factor: RealValueOrNumber,
  ) {
    super();
  }
}

export class Union extends ProfileNode {
  public readonly nodeType = "Union";
  constructor(
    public readonly left: ProfileNode,
    public readonly right: ProfileNode,
  ) {
    super();
  }
}

export class SmoothUnion extends ProfileNode {
  public readonly nodeType = "SmoothUnion";
  constructor(
    public readonly left: ProfileNode,
    public readonly right: ProfileNode,
    public readonly radius: DistanceNode,
  ) {
    super();
  }
}

export class Intersection extends ProfileNode {
  public readonly nodeType = "Intersection";
  constructor(
    public readonly left: ProfileNode,
    public readonly right: ProfileNode,
  ) {
    super();
  }
}

export class SmoothIntersection extends ProfileNode {
  public readonly nodeType = "SmoothIntersection";
  constructor(
    public readonly left: ProfileNode,
    public readonly right: ProfileNode,
    public readonly radius: DistanceNode,
  ) {
    super();
  }
}

export class Difference extends ProfileNode {
  public readonly nodeType = "Difference";
  constructor(
    public readonly left: ProfileNode,
    public readonly right: ProfileNode,
  ) {
    super();
  }
}

export class SmoothDifference extends ProfileNode {
  public readonly nodeType = "SmoothDifference";
  constructor(
    public readonly left: ProfileNode,
    public readonly right: ProfileNode,
    public readonly radius: DistanceNode,
  ) {
    super();
  }
}

export class Shell extends ProfileNode {
  public readonly nodeType = "Shell";
  constructor(
    public readonly profile: ProfileNode,
    public readonly thickness: DistanceNode,
  ) {
    super();
  }
}

export class Morph extends ProfileNode {
  public readonly nodeType = "Morph";
  constructor(
    public readonly start: ProfileNode,
    public readonly end: ProfileNode,
    public readonly t: DistanceNode,
  ) {
    super();
  }
}

export class Dilate extends ProfileNode {
  public readonly nodeType = "Dilate";
  constructor(
    public readonly profile: ProfileNode,
    public readonly factor: DistanceNode,
  ) {
    super();
  }
}

export class FlipNode extends ProfileNode {
  public readonly nodeType = "Flip";
  constructor(
    public readonly profile: ProfileNode,
    public readonly axis: "x" | "y",
  ) {
    super();
  }
}

export class NormalizedFieldNode extends ProfileNode {
  public readonly nodeType = "NormalizedField";
  constructor(public readonly profile: ProfileNode) {
    super();
  }
}

export class MidSurfaceNode extends ProfileNode {
  public readonly nodeType = "MidSurface";
  constructor(
    public readonly first: ProfileNode,
    public readonly second: ProfileNode,
  ) {
    super();
  }
}

export type AnyProfileOperationNode =
  | Translation
  | Rotation
  | Scale
  | Union
  | SmoothUnion
  | Intersection
  | SmoothIntersection
  | Difference
  | SmoothDifference
  | Shell
  | Morph
  | Dilate
  | FlipNode
  | NormalizedFieldNode
  | MidSurfaceNode;
