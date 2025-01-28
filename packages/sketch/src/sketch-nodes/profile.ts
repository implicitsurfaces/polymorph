import {
  AngleNode,
  DistanceNode,
  EdgeNode,
  PathNode,
  PlaneNode,
  ProfileNode,
} from "./bases";

export class PathClose extends ProfileNode {
  constructor(
    public readonly path: PathNode,
    public readonly edge: EdgeNode,
  ) {
    super();
  }
}

export class PathOpenEnd extends ProfileNode {
  constructor(public readonly path: PathNode) {
    super();
  }
}

export class Circle extends ProfileNode {
  constructor(public readonly radius: DistanceNode) {
    super();
  }
}

export class Box extends ProfileNode {
  constructor(
    public readonly width: DistanceNode,
    public readonly height: DistanceNode,
  ) {
    super();
  }
}

export class SolidSliceNode extends ProfileNode {
  constructor(
    public readonly solid: ProfileNode,
    public readonly plane: PlaneNode,
  ) {
    super();
  }
}

export class EllipseNode extends ProfileNode {
  constructor(
    public readonly majorRadius: DistanceNode,
    public readonly minorRadius: DistanceNode,
  ) {
    super();
  }
}

export class EllipseArcNode extends ProfileNode {
  constructor(
    public readonly majorRadius: DistanceNode,
    public readonly minorRadius: DistanceNode,
    public readonly startAngle: AngleNode,
    public readonly endAngle: AngleNode,
    public readonly orientation: 1 | -1,
  ) {
    super();
  }
}
