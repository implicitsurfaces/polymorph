import {
  AngleNode,
  DistanceNode,
  ProfileNode,
  RealValueNode,
  VectorNode,
} from "../sketch-nodes";

export class Translation extends ProfileNode {
  constructor(
    public readonly profile: ProfileNode,
    public readonly vector: VectorNode,
  ) {
    super();
  }
}

export class Rotation extends ProfileNode {
  constructor(
    public readonly profile: ProfileNode,
    public readonly angle: AngleNode,
  ) {
    super();
  }
}

export class Scale extends ProfileNode {
  constructor(
    public readonly profile: ProfileNode,
    public readonly factor: RealValueNode,
  ) {
    super();
  }
}

export class Union extends ProfileNode {
  constructor(
    public readonly left: ProfileNode,
    public readonly right: ProfileNode,
  ) {
    super();
  }
}

export class SmoothUnion extends ProfileNode {
  constructor(
    public readonly left: ProfileNode,
    public readonly right: ProfileNode,
    public readonly radius: DistanceNode,
  ) {
    super();
  }
}

export class Intersection extends ProfileNode {
  constructor(
    public readonly left: ProfileNode,
    public readonly right: ProfileNode,
  ) {
    super();
  }
}

export class SmoothIntersection extends ProfileNode {
  constructor(
    public readonly left: ProfileNode,
    public readonly right: ProfileNode,
    public readonly radius: DistanceNode,
  ) {
    super();
  }
}

export class Difference extends ProfileNode {
  constructor(
    public readonly left: ProfileNode,
    public readonly right: ProfileNode,
  ) {
    super();
  }
}

export class SmoothDifference extends ProfileNode {
  constructor(
    public readonly left: ProfileNode,
    public readonly right: ProfileNode,
    public readonly radius: DistanceNode,
  ) {
    super();
  }
}

export class Shell extends ProfileNode {
  constructor(
    public readonly profile: ProfileNode,
    public readonly thickness: DistanceNode,
  ) {
    super();
  }
}

export class Morph extends ProfileNode {
  constructor(
    public readonly start: ProfileNode,
    public readonly end: ProfileNode,
    public readonly t: DistanceNode,
  ) {
    super();
  }
}

export class Dilate extends ProfileNode {
  constructor(
    public readonly profile: ProfileNode,
    public readonly factor: DistanceNode,
  ) {
    super();
  }
}

export class FlipNode extends ProfileNode {
  constructor(
    public readonly profile: ProfileNode,
    public readonly axis: "x" | "y",
  ) {
    super();
  }
}

export class NormalizedFieldNode extends ProfileNode {
  constructor(public readonly profile: ProfileNode) {
    super();
  }
}

export class MidSurfaceNode extends ProfileNode {
  constructor(
    public readonly first: ProfileNode,
    public readonly second: ProfileNode,
  ) {
    super();
  }
}
