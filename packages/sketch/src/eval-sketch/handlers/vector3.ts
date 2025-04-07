import { Point3D, Vec3 } from "../../geom-3d";
import { variable } from "../../num";
import { gradientAt } from "../../num-ops";
import {
  AnyVector3Node,
  SolidGradientAt,
  Vector3Difference,
  Vector3FromCartesianCoords,
  Vector3FromPoint,
  Vector3FromPoints,
  Vector3Rotated,
  Vector3Scaled,
  Vector3Sum,
} from "../../sketch-nodes";
import { Handler } from "../../sketch-nodes/types";
import {
  guardAngle,
  guardNum,
  guardPoint3,
  guardSolid,
  guardVec3,
} from "../guards";
import { getAxis } from "./utils/getAxis";

export interface Vector3NodeHandler<T extends AnyVector3Node>
  extends Handler<T> {
  category: "Vector3";
  eval: (vector: T, children: unknown[]) => Vec3;
}

const vector3FromPointHandler: Vector3NodeHandler<Vector3FromPoint> = {
  category: "Vector3",
  nodeType: "Vector3FromPoint",
  children: (node) => [node.point],
  eval: function evalVector3FromPoint(_, [point]) {
    return guardPoint3(point).vecFromOrigin();
  },
};

const vector3FromPointsHandler: Vector3NodeHandler<Vector3FromPoints> = {
  category: "Vector3",
  nodeType: "Vector3FromPoints",
  children: (node) => [node.p0, node.p1],
  eval: function evalVector3FromPoints(_, [p0, p1]) {
    return guardPoint3(p0).vecTo(guardPoint3(p1));
  },
};

const vector3FromCartesianCoordsHandler: Vector3NodeHandler<Vector3FromCartesianCoords> =
  {
    category: "Vector3",
    nodeType: "Vector3FromCartesianCoords",
    children: (node) => [node.x, node.y, node.z],
    eval: function evalVector3FromCartesianCoords(_, [x, y, z]) {
      return new Vec3(guardNum(x), guardNum(y), guardNum(z));
    },
  };

const vector3SumHandler: Vector3NodeHandler<Vector3Sum> = {
  category: "Vector3",
  nodeType: "Vector3Sum",
  children: (node) => [node.left, node.right],
  eval: function evalVector3Sum(_, [left, right]) {
    return guardVec3(left).add(guardVec3(right));
  },
};

const vector3DifferenceHandler: Vector3NodeHandler<Vector3Difference> = {
  category: "Vector3",
  nodeType: "Vector3Difference",
  children: (node) => [node.left, node.right],
  eval: function evalVector3Difference(_, [left, right]) {
    return guardVec3(left).sub(guardVec3(right));
  },
};

const vector3ScaledHandler: Vector3NodeHandler<Vector3Scaled> = {
  category: "Vector3",
  nodeType: "Vector3Scaled",
  children: (node) => [node.vector, node.scale],
  eval: function evalVector3Scaled(_, [vector, scale]) {
    return guardVec3(vector).scale(guardNum(scale));
  },
};

const vector3RotatedHandler: Vector3NodeHandler<Vector3Rotated> = {
  category: "Vector3",
  nodeType: "Vector3Rotated",
  children: (node) => {
    if (node.axis === "x" || node.axis === "y" || node.axis === "z") {
      return [node.vector, node.angle];
    }
    return [node.vector, node.angle, node.axis];
  },
  eval: function evalVector3Rotated(node, [vector, angle, inputAxis]) {
    const axis = getAxis(node.axis, inputAxis);
    return guardVec3(vector).rotate(guardAngle(angle), axis);
  },
};

const solidGradientAtHandler: Vector3NodeHandler<SolidGradientAt> = {
  category: "Vector3",
  nodeType: "SolidGradientAt",
  children: (node) => [node.field, node.point],
  eval: function evalSolidGradientAt(_, [solid, inputPoint]) {
    const point = guardPoint3(inputPoint);
    const p = new Point3D(variable("!!x"), variable("!!y"), variable("!!z"));
    const grad = gradientAt(guardSolid(solid).valueAt(p), [
      ["!!x", point.x],
      ["!!y", point.y],
      ["!!z", point.z],
    ]);
    return new Vec3(grad[0], grad[1], grad[2]);
  },
};

export const vector3Handlers = [
  vector3FromCartesianCoordsHandler,
  vector3FromPointHandler,
  vector3FromPointsHandler,
  vector3SumHandler,
  vector3DifferenceHandler,
  vector3ScaledHandler,
  vector3RotatedHandler,
  solidGradientAtHandler,
] as const;
