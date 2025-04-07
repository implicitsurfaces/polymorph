import { Num, variable } from "../../num";
import {
  AnySimpleRealValueNode,
  RealValueVariable,
  SignedDistanceToProfile,
} from "../../sketch-nodes";
import { Handler } from "../../sketch-nodes/types";
import { guardDistField, guardPoint } from "../guards";

export interface RealValueHandler<T extends AnySimpleRealValueNode>
  extends Handler<T> {
  category: "RealValue";
  eval: (value: T, children: unknown[]) => Num;
}

const realValueVariableHandler: RealValueHandler<RealValueVariable> = {
  category: "RealValue",
  nodeType: "RealValueVariable",
  children: () => [],
  eval: (node) => variable(node.name),
};

const signedDistanceToProfileHandler: RealValueHandler<SignedDistanceToProfile> =
  {
    category: "RealValue",
    nodeType: "SignedDistanceToProfile",
    children: (node) => [node.profile, node.point],
    eval: function evalSignedDistanceToProfile(_, [profile, point]) {
      return guardDistField(profile).distanceTo(guardPoint(point));
    },
  };

export const realValueHandlers = [
  realValueVariableHandler,
  signedDistanceToProfileHandler,
] as const;
