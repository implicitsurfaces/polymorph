import { asNum, Num, variable } from "../../num";
import { sigmoid } from "../../num-ops";
import {
  AnyDistanceNode,
  ArcLength,
  DistanceLiteral,
  DistanceScaled,
  DistanceSum,
  DistanceVariable,
  DistanceFromReal,
  Vector3Norm,
  VectorNorm,
} from "../../sketch-nodes";
import { Handler } from "../../sketch-nodes/types";
import { guardAngle, guardNum, guardVec2, guardVec3 } from "../guards";

export interface DistanceHandler<T extends AnyDistanceNode = AnyDistanceNode>
  extends Handler<T> {
  category: "Distance";
  eval: (distance: T, children: unknown[]) => Num;
}

const distanceLiteralHandler: DistanceHandler<DistanceLiteral> = {
  category: "Distance",
  nodeType: "DistanceLiteral",
  children: () => [],
  eval: (node) => asNum(node.value),
};

const distanceFromRealValueHandler: DistanceHandler<DistanceFromReal> = {
  category: "Distance",
  nodeType: "DistanceFromReal",
  children: (node) => [node.value],
  eval: function evalDistanceFromRealValue(_, [realValue]) {
    return guardNum(realValue).abs();
  },
};

const distanceVariableHandler: DistanceHandler<DistanceVariable> = {
  category: "Distance",
  nodeType: "DistanceVariable",
  children: () => [],
  eval: (node) => {
    const v = variable(node.name);

    if (node.max !== undefined) {
      const a = asNum(node.min ?? 0);
      const b = asNum(node.max);

      return a.add(b.sub(a).mul(sigmoid(v)));
    }

    if (node.min !== undefined) {
      return asNum(node.min).add(v.exp());
    }

    return v.exp();
  },
};

const distanceSumHandler: DistanceHandler<DistanceSum> = {
  category: "Distance",
  nodeType: "DistanceSum",
  children: (node) => [node.left, node.right],
  eval: function evalDistanceSum(_, [left, right]) {
    return guardNum(left).add(guardNum(right));
  },
};

const distanceScaledHandler: DistanceHandler<DistanceScaled> = {
  category: "Distance",
  nodeType: "DistanceScaled",
  children: (node) => [node.distance, node.scale],
  eval: function evalDistanceScaled(_, [distance, scale]) {
    return guardNum(distance).mul(guardNum(scale));
  },
};

const vectorNormHandler: DistanceHandler<VectorNorm> = {
  category: "Distance",
  nodeType: "VectorNorm",
  children: (node) => [node.vector],
  eval: function evalVectorNorm(_, [vector]) {
    return guardVec2(vector).norm();
  },
};

const vector3NormHandler: DistanceHandler<Vector3Norm> = {
  category: "Distance",
  nodeType: "Vector3Norm",
  children: (node) => [node.vector],
  eval: function evalVector3Norm(_, [vector]) {
    return guardVec3(vector).norm();
  },
};

const arcLengthHandler: DistanceHandler<ArcLength> = {
  category: "Distance",
  nodeType: "ArcLength",
  children: (node) => [node.angle, node.radius],
  eval: function evalArcLength(_, [angle, radius]) {
    return guardAngle(angle).asRad().mul(guardNum(radius));
  },
};

export const distanceHandlers = [
  distanceLiteralHandler,
  distanceVariableHandler,
  distanceSumHandler,
  distanceScaledHandler,
  vectorNormHandler,
  vector3NormHandler,
  arcLengthHandler,
  distanceFromRealValueHandler,
] as const;
