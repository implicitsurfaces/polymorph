import { PointNode, ProfileNode, RealValueNode } from "./bases";

export class RealValueVariable extends RealValueNode {
  public readonly nodeType = "RealValueVariable";
  constructor(public readonly name: string) {
    super();
  }
}

export class SignedDistanceToProfile extends RealValueNode {
  public readonly nodeType = "SignedDistanceToProfile";
  constructor(
    public readonly profile: ProfileNode,
    public readonly point: PointNode,
  ) {
    super();
  }
}

export type AnySimpleRealValueNode =
  | RealValueVariable
  | SignedDistanceToProfile;
