import { Angle, angleFromDeg, angleFromSin } from "../../geom";
import { variable } from "../../num";
import { sigmoid } from "../../num-ops";
import {
  AngleBisection,
  AngleDifference,
  AngleLiteral,
  AngleOpposite,
  AnglePerpendicular,
  AngleSum,
  AngleVariable,
  AnyAngleNode,
  VectorDirection,
} from "../../sketch-nodes";
import { Handler } from "../../sketch-nodes/types";
import { guardAngle, guardNum, guardVec2 } from "../guards";

export interface AngleHandler<Node extends AnyAngleNode> extends Handler<Node> {
  category: "Angle";
  eval: (node: Node, children: unknown[]) => Angle;
}

const angleLiteralHandler: AngleHandler<AngleLiteral> = {
  category: "Angle",
  nodeType: "AngleLiteral",
  children: (node) => [node.degrees],
  eval: function evalAngleLiteral(_, [degrees]) {
    return angleFromDeg(guardNum(degrees));
  },
};

const angleVariableHandler: AngleHandler<AngleVariable> = {
  category: "Angle",
  nodeType: "AngleVariable",
  children: () => [],
  eval: function evalAngleVariable(node) {
    const v = variable(node.name);
    return angleFromSin(sigmoid(v).mul(2).sub(1)).double();
  },
};

const angleSumHandler: AngleHandler<AngleSum> = {
  category: "Angle",
  nodeType: "AngleSum",
  children: (node) => [node.left, node.right],
  eval: function evalAngleSum(_, [left, right]) {
    return guardAngle(left).add(guardAngle(right));
  },
};

const angleDifferenceHandler: AngleHandler<AngleDifference> = {
  category: "Angle",
  nodeType: "AngleDifference",
  children: (node) => [node.left, node.right],
  eval: function evalAngleDifference(_, [left, right]) {
    return guardAngle(left).sub(guardAngle(right));
  },
};

const anglePerpendicularHandler: AngleHandler<AnglePerpendicular> = {
  category: "Angle",
  nodeType: "AnglePerpendicular",
  children: (node) => [node.angle],
  eval: function evalAnglePerpendicular(_, [angle]) {
    return guardAngle(angle).perp();
  },
};

const angleOppositeHandler: AngleHandler<AngleOpposite> = {
  category: "Angle",
  nodeType: "AngleOpposite",
  children: (node) => [node.angle],
  eval: function evalAngleOpposite(_, [angle]) {
    return guardAngle(angle).opposite();
  },
};

const angleBisectionHandler: AngleHandler<AngleBisection> = {
  category: "Angle",
  nodeType: "AngleBisection",
  children: (node) => [node.angle],
  eval: function evalAngleBisection(_, [angle]) {
    return guardAngle(angle).half();
  },
};

const vectorDirectionHandler: AngleHandler<VectorDirection> = {
  category: "Angle",
  nodeType: "VectorDirection",
  children: (node) => [node.vector],
  eval: function evalVectorDirection(_, [vector]) {
    return guardVec2(vector).asAngle();
  },
};

export const angleHandlers = [
  angleLiteralHandler,
  angleVariableHandler,
  angleSumHandler,
  angleDifferenceHandler,
  anglePerpendicularHandler,
  angleOppositeHandler,
  angleBisectionHandler,
  vectorDirectionHandler,
];
