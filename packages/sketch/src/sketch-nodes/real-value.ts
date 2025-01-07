import { PointNode, ProfileNode, RealValueNode } from "../sketch-nodes";

export class RealValueVariable extends RealValueNode {
  constructor(public readonly name: string) {
    super();
  }
}

export class SignedDistanceToProfile extends RealValueNode {
  constructor(
    public readonly profile: ProfileNode,
    public readonly point: PointNode,
  ) {
    super();
  }
}
