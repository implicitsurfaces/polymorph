import { Point } from "../../geom";
import {
  AnyPointNode,
  PointAsVectorFromOrigin,
  PointMidPoint,
  PointVectorDifference,
  PointVectorSum,
} from "../../sketch-nodes";
import { Handler } from "../../sketch-nodes/types";
import { guardPoint, guardVec2 } from "../guards";

export interface PointHandler<T extends AnyPointNode> extends Handler<T> {
  category: "Point";
  eval: (point: T, children: unknown[]) => Point;
}

const pointAsVectorFromOriginHandler: PointHandler<PointAsVectorFromOrigin> = {
  category: "Point",
  nodeType: "PointAsVectorFromOrigin",
  children: (node) => [node.vector],
  eval: function evalPointAsVectorFromOrigin(_, [vector]) {
    return guardVec2(vector).pointFromOrigin();
  },
};

const pointVectorSumHandler: PointHandler<PointVectorSum> = {
  category: "Point",
  nodeType: "PointVectorSum",
  children: (node) => [node.point, node.vector],
  eval: function evalPointVectorSum(_, [point, vector]) {
    return guardPoint(point).add(guardVec2(vector));
  },
};

const pointVectorDifferenceHandler: PointHandler<PointVectorDifference> = {
  category: "Point",
  nodeType: "PointVectorDifference",
  children: (node) => [node.point, node.vector],
  eval: function evalPointVectorDifference(_, [point, vector]) {
    return guardPoint(point).sub(guardVec2(vector));
  },
};

const pointMidPointHandler: PointHandler<PointMidPoint> = {
  category: "Point",
  nodeType: "PointMidPoint",
  children: (node) => [node.left, node.right],
  eval: function evalPointMidPoint(_, [left, right]) {
    return guardPoint(left).midPoint(guardPoint(right));
  },
};

export const pointHandlers = [
  pointAsVectorFromOriginHandler,
  pointVectorSumHandler,
  pointVectorDifferenceHandler,
  pointMidPointHandler,
] as const;
