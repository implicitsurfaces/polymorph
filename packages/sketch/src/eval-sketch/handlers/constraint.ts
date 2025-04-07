import { Num, ONE, TWO } from "../../num";
import {
  AnyConstraintNode,
  ConstraintOnAngle,
  ConstraintOnDistance,
  ConstraintOnPoint,
  ConstraintOnProfileBoundary,
} from "../../sketch-nodes";
import { Handler } from "../../sketch-nodes/types";
import { guardAngle, guardNum, guardPoint, guardProfile } from "../guards";

export interface ConstraintNodeHandler<T extends AnyConstraintNode>
  extends Handler<T> {
  category: "Constraint";
  eval: (constraint: T, children: unknown[]) => Num;
}

const constraintOnDistanceHandler: ConstraintNodeHandler<ConstraintOnDistance> =
  {
    category: "Constraint",
    nodeType: "ConstraintOnDistance",
    children: (node) => [node.distance, node.target, node.weigth ?? 1],
    eval: function evalConstraintOnDistance(_, [distance, target, weigth]) {
      return guardNum(distance)
        .sub(guardNum(target))
        .div(guardNum(weigth))
        .square();
    },
  };

const constraintOnAngleHandler: ConstraintNodeHandler<ConstraintOnAngle> = {
  category: "Constraint",
  nodeType: "ConstraintOnAngle",
  children: (node) => [node.angle, node.target, node.weigth ?? 1],
  eval: function evalConstraintOnAngle(
    _,
    [angleInput, targetInput, weigthInput],
  ) {
    const angle = guardAngle(angleInput);
    const target = guardAngle(targetInput);
    const tol = guardNum(weigthInput);

    const loss = TWO.sub(angle.sub(target).cos().add(ONE));
    return loss.div(tol.mul(TWO));
  },
};

const constraintOnPointHandler: ConstraintNodeHandler<ConstraintOnPoint> = {
  category: "Constraint",
  nodeType: "ConstraintOnPoint",
  children: (node) => [node.point, node.target, node.weigth ?? 1],
  eval: function evalConstraintOnPoint(
    _,
    [pointInput, targetInput, weigthInput],
  ) {
    const point = guardPoint(pointInput);
    const target = guardPoint(targetInput);
    const tol = guardNum(weigthInput);

    const diff = point.vecFrom(target).norm();
    return diff.div(tol);
  },
};

const constraintOnProfileBoundaryHandler: ConstraintNodeHandler<ConstraintOnProfileBoundary> =
  {
    category: "Constraint",
    nodeType: "ConstraintOnProfileBoundary",
    children: (node) => [
      node.profile,
      node.point,
      node.signedDistance ?? 0,
      node.weigth ?? 1,
    ],
    eval: function evalConstraintOnProfileBoundary(
      _,
      [profileInput, pointInput, signedDistanceInput, weigthInput],
    ) {
      const profile = guardProfile(profileInput);
      const point = guardPoint(pointInput);
      const signedDistanceNum = guardNum(signedDistanceInput);
      const weigthNum = guardNum(weigthInput);

      const dist = profile.distanceTo(point);
      return dist.sub(signedDistanceNum).div(weigthNum).square();
    },
  };

export const constraintHandlers = [
  constraintOnDistanceHandler,
  constraintOnAngleHandler,
  constraintOnPointHandler,
  constraintOnProfileBoundaryHandler,
];
