export class RealValueNode {
  public readonly category = "RealValue";
}

export class DistanceNode {
  public readonly category = "Distance";
}

export type RealValueOrNumber = number | RealValueNode | DistanceNode;

export class AngleNode {
  public readonly category = "Angle";
}

export class PointNode {
  public readonly category = "Point";
}

export class Point3Node {
  public readonly category = "Point3";
}

export class VectorNode {
  public readonly category = "Vector";
}

export class Vector3Node {
  public readonly category = "Vector3";
}

export class PlaneNode {
  public readonly category = "Plane";
}

export class EdgeNode {
  public readonly category = "Edge";
}

export class PathNode {
  public readonly category = "Path";
}

export class ProfileNode {
  public readonly category = "Profile";
}

export class SolidNode {
  public readonly category = "Solid";
}

export class WidthModulationNode {
  public readonly category = "WidthModulation";
}

export class ConstraintNode {
  public readonly category = "Constraint";
}
