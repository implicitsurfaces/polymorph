import { Point, Vec2 } from "../../geom";
import { variable } from "../../num";
import { gradientAt } from "../../num-ops";
import {
  AnyVectorNode,
  GradientAt,
  VectorDifference,
  VectorFromCartesianCoords,
  VectorFromPoint,
  VectorFromPoints,
  VectorFromPolarCoods,
  VectorRotated,
  VectorScaled,
  VectorSum,
} from "../../sketch-nodes";
import { Handler } from "../../sketch-nodes/types";
import {
  guardAngle,
  guardNum,
  guardPoint,
  guardProfile,
  guardVec2,
} from "../guards";

export interface VectorHandler<T extends AnyVectorNode> extends Handler<T> {
  category: "Vector";
  eval: (vector: T, children: unknown[]) => Vec2;
}

export const vectorFromCartesianCoordsHandler: VectorHandler<VectorFromCartesianCoords> =
  {
    category: "Vector",
    nodeType: "VectorFromCartesianCoords",
    children: (node) => [node.x, node.y],
    eval: function evalVectorFromCartesianCoords(_, [x, y]) {
      return new Vec2(guardNum(x), guardNum(y));
    },
  };

export const vectorFromPolarCoodsHandler: VectorHandler<VectorFromPolarCoods> =
  {
    category: "Vector",
    nodeType: "VectorFromPolarCoods",
    children: (node) => [node.distance, node.angle],
    eval: function evalVectorFromPolarCoods(_, [distance, angle]) {
      return guardAngle(angle).asVec().scale(guardNum(distance));
    },
  };

export const vectorFromPointHandler: VectorHandler<VectorFromPoint> = {
  category: "Vector",
  nodeType: "VectorFromPoint",
  children: (node) => [node.point],
  eval: function evalVectorFromPoint(_, [point]) {
    return guardPoint(point).vecFromOrigin();
  },
};

export const vectorFromPointsHandler: VectorHandler<VectorFromPoints> = {
  category: "Vector",
  nodeType: "VectorFromPoints",
  children: (node) => [node.p0, node.p1],
  eval: function evalVectorFromPoints(_, [p0, p1]) {
    return guardPoint(p0).vecTo(guardPoint(p1));
  },
};

export const vectorSumHandler: VectorHandler<VectorSum> = {
  category: "Vector",
  nodeType: "VectorSum",
  children: (node) => [node.left, node.right],
  eval: function evalVectorSum(_, [left, right]) {
    return guardVec2(left).add(guardVec2(right));
  },
};

export const vectorDifferenceHandler: VectorHandler<VectorDifference> = {
  category: "Vector",
  nodeType: "VectorDifference",
  children: (node) => [node.left, node.right],
  eval: function evalVectorDifference(_, [left, right]) {
    return guardVec2(left).sub(guardVec2(right));
  },
};

export const vectorScaleHandler: VectorHandler<VectorScaled> = {
  category: "Vector",
  nodeType: "VectorScaled",
  children: (node) => [node.vector, node.scale],
  eval: function evalVectorScale(_, [vector, scale]) {
    return guardVec2(vector).scale(guardNum(scale));
  },
};

export const vectorRotatedHandler: VectorHandler<VectorRotated> = {
  category: "Vector",
  nodeType: "VectorRotated",
  children: (node) => [node.vector, node.angle],
  eval: function evalVectorRotated(_, [vector, angle]) {
    return guardVec2(vector).rotate(guardAngle(angle));
  },
};

export const gradientAtHandler: VectorHandler<GradientAt> = {
  category: "Vector",
  nodeType: "GradientAt",
  children: (node) => [node.field, node.point],
  eval: function evalGradientAt(_, [field, inputPoint]) {
    const p = new Point(variable("!!x"), variable("!!y"));
    const point = guardPoint(inputPoint);

    const grad = gradientAt(guardProfile(field).distanceTo(p), [
      ["!!x", point.x],
      ["!!y", point.y],
    ]);
    return new Vec2(grad[0], grad[1]);
  },
};

export const vectorHandlers = [
  vectorFromCartesianCoordsHandler,
  vectorFromPolarCoodsHandler,
  vectorFromPointHandler,
  vectorFromPointsHandler,
  vectorSumHandler,
  vectorDifferenceHandler,
  vectorScaleHandler,
  vectorRotatedHandler,
  gradientAtHandler,
] as const;
