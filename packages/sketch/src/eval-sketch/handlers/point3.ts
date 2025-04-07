import { Point3D } from "../../geom-3d";
import {
  AnyPoint3Node,
  Point3AsVectorFromOrigin,
  Point3MidPoint,
  Point3VectorDifference,
  Point3VectorSum,
} from "../../sketch-nodes";
import { Handler } from "../../sketch-nodes/types";
import { guardVec3, guardPoint3 } from "../guards";

export interface Point3DHandler<T extends AnyPoint3Node> extends Handler<T> {
  category: "Point3";
  eval: (point: T, children: unknown[]) => Point3D;
}

const point3AsVectorFromOriginHandler: Point3DHandler<Point3AsVectorFromOrigin> =
  {
    category: "Point3",
    nodeType: "Point3AsVectorFromOrigin",
    children: (node) => [node.vector],
    eval: function evalPoint3AsVectorFromOrigin(_, [vector]) {
      return guardVec3(vector).pointFromOrigin();
    },
  };

const point3VectorSumHandler: Point3DHandler<Point3VectorSum> = {
  category: "Point3",
  nodeType: "Point3VectorSum",
  children: (node) => [node.point, node.vector],
  eval: function evalPoint3VectorSum(_, [point, vector]) {
    return guardPoint3(point).add(guardVec3(vector));
  },
};

const point3VectorDifferenceHandler: Point3DHandler<Point3VectorDifference> = {
  category: "Point3",
  nodeType: "Point3VectorDifference",
  children: (node) => [node.point, node.vector],
  eval: function evalPoint3VectorDifference(_, [point, vector]) {
    return guardPoint3(point).sub(guardVec3(vector));
  },
};

const point3MidPointHandler: Point3DHandler<Point3MidPoint> = {
  category: "Point3",
  nodeType: "Point3MidPoint",
  children: (node) => [node.left, node.right],
  eval: function evalPoint3MidPoint(_, [left, right]) {
    return guardPoint3(left).midPoint(guardPoint3(right));
  },
};

export const point3Handlers = [
  point3AsVectorFromOriginHandler,
  point3VectorSumHandler,
  point3VectorDifferenceHandler,
  point3MidPointHandler,
] as const;
