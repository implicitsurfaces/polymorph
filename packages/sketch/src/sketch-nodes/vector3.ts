import {
  AngleNode,
  PointNode,
  RealValueNode,
  SolidNode,
  Point3Node,
  Vector3Node,
} from "./bases";

export class Vector3FromCartesianCoords extends Vector3Node {
  constructor(
    public readonly x: RealValueNode,
    public readonly y: RealValueNode,
    public readonly z: RealValueNode,
  ) {
    super();
  }
}

export class Vector3FromPoints extends Vector3Node {
  constructor(
    public readonly p0: Point3Node,
    public readonly p1: Point3Node,
  ) {
    super();
  }
}

export class Vector3FromPoint extends Vector3Node {
  constructor(public readonly point: PointNode) {
    super();
  }
}

export class Vector3Sum extends Vector3Node {
  constructor(
    public readonly left: Vector3Node,
    public readonly right: Vector3Node,
  ) {
    super();
  }
}

export class Vector3Difference extends Vector3Node {
  constructor(
    public readonly left: Vector3Node,
    public readonly right: Vector3Node,
  ) {
    super();
  }
}

export class Vector3Scaled extends Vector3Node {
  constructor(
    public readonly vector: Vector3Node,
    public readonly scale: RealValueNode,
  ) {
    super();
  }
}

export class Vector3Rotated extends Vector3Node {
  constructor(
    public readonly vector: Vector3Node,
    public readonly angle: AngleNode,
    public readonly axis: "x" | "y" | "z" | Vector3Node,
  ) {
    super();
  }
}

export class SolidGradientAt extends Vector3Node {
  constructor(
    public readonly field: SolidNode,
    public readonly point: Point3Node,
  ) {
    super();
  }
}
