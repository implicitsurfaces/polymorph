import { AngleNode, PlaneNode, Point3Node, Vector3Node } from "./bases";

export class BasePlaneNode extends PlaneNode {
  constructor(public readonly plane: "xy" | "yz" | "xz") {
    super();
  }
}

export class TranslatedPlaneNode extends PlaneNode {
  constructor(
    public readonly plane: PlaneNode,
    public readonly vector: Vector3Node,
  ) {
    super();
  }
}

export class PivotedPlaneNode extends PlaneNode {
  constructor(
    public readonly plane: PlaneNode,
    public readonly angle: AngleNode,
    public readonly axis: "x" | "y" | "z" | Vector3Node,
  ) {
    super();
  }
}

export class RotatedPlaneNode extends PlaneNode {
  constructor(
    public readonly plane: PlaneNode,
    public readonly angle: AngleNode,
  ) {
    super();
  }
}

export class PlaneFromPoints extends PlaneNode {
  constructor(
    public readonly origin: Point3Node,
    public readonly p1: Point3Node,
    public readonly p2: Point3Node,
  ) {
    super();
  }
}
