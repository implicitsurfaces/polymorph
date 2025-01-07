import { PointNode, ProfileNode, RealValueNode } from "./bases";

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
