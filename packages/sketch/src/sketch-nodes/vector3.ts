import {
  AngleNode,
  RealValueOrNumber,
  SolidNode,
  Point3Node,
  Vector3Node,
} from "./bases";

export class Vector3FromCartesianCoords extends Vector3Node {
  public readonly nodeType = "Vector3FromCartesianCoords";
  constructor(
    public readonly x: RealValueOrNumber,
    public readonly y: RealValueOrNumber,
    public readonly z: RealValueOrNumber,
  ) {
    super();
  }
}

export class Vector3FromPoints extends Vector3Node {
  public readonly nodeType = "Vector3FromPoints";
  constructor(
    public readonly p0: Point3Node,
    public readonly p1: Point3Node,
  ) {
    super();
  }
}

export class Vector3FromPoint extends Vector3Node {
  public readonly nodeType = "Vector3FromPoint";
  constructor(public readonly point: Point3Node) {
    super();
  }
}

export class Vector3Sum extends Vector3Node {
  public readonly nodeType = "Vector3Sum";
  constructor(
    public readonly left: Vector3Node,
    public readonly right: Vector3Node,
  ) {
    super();
  }
}

export class Vector3Difference extends Vector3Node {
  public readonly nodeType = "Vector3Difference";
  constructor(
    public readonly left: Vector3Node,
    public readonly right: Vector3Node,
  ) {
    super();
  }
}

export class Vector3Scaled extends Vector3Node {
  public readonly nodeType = "Vector3Scaled";
  constructor(
    public readonly vector: Vector3Node,
    public readonly scale: RealValueOrNumber,
  ) {
    super();
  }
}

export class Vector3Rotated extends Vector3Node {
  public readonly nodeType = "Vector3Rotated";
  constructor(
    public readonly vector: Vector3Node,
    public readonly angle: AngleNode,
    public readonly axis: "x" | "y" | "z" | Vector3Node,
  ) {
    super();
  }
}

export class SolidGradientAt extends Vector3Node {
  public readonly nodeType = "SolidGradientAt";
  constructor(
    public readonly field: SolidNode,
    public readonly point: Point3Node,
  ) {
    super();
  }
}

export type AnyVector3Node =
  | Vector3FromCartesianCoords
  | Vector3FromPoints
  | Vector3FromPoint
  | Vector3Sum
  | Vector3Difference
  | Vector3Scaled
  | Vector3Rotated
  | SolidGradientAt;
