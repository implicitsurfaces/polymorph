import { Point3Node, SolidNode, Vector3Node } from "./bases";

export class Point3VectorSum extends Point3Node {
  constructor(
    public readonly point: Point3Node,
    public readonly vector: Vector3Node,
  ) {
    super();
  }
}

export class Point3VectorDifference extends Point3Node {
  constructor(
    public readonly point: Point3Node,
    public readonly vector: Vector3Node,
  ) {
    super();
  }
}

export class Point3MidPoint extends Point3Node {
  constructor(
    public readonly left: Point3Node,
    public readonly right: Point3Node,
  ) {
    super();
  }
}

export class Point3AsVectorFromOrigin extends Point3Node {
  constructor(public readonly vector: Vector3Node) {
    super();
  }
}

export class SolidCentroid extends Point3Node {
  constructor(public readonly profile: SolidNode) {
    super();
  }
}
