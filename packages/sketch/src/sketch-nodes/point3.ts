import { Point3Node, Vector3Node } from "./bases";

export class Point3VectorSum extends Point3Node {
  public readonly nodeType = "Point3VectorSum";
  constructor(
    public readonly point: Point3Node,
    public readonly vector: Vector3Node,
  ) {
    super();
  }
}

export class Point3VectorDifference extends Point3Node {
  public readonly nodeType = "Point3VectorDifference";
  constructor(
    public readonly point: Point3Node,
    public readonly vector: Vector3Node,
  ) {
    super();
  }
}

export class Point3MidPoint extends Point3Node {
  public readonly nodeType = "Point3MidPoint";
  constructor(
    public readonly left: Point3Node,
    public readonly right: Point3Node,
  ) {
    super();
  }
}

export class Point3AsVectorFromOrigin extends Point3Node {
  public readonly nodeType = "Point3AsVectorFromOrigin";
  constructor(public readonly vector: Vector3Node) {
    super();
  }
}

export type AnyPoint3Node =
  | Point3VectorSum
  | Point3VectorDifference
  | Point3MidPoint
  | Point3AsVectorFromOrigin;
