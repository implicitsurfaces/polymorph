import { Plane, UnitVec3, XY_PLANE, XZ_PLANE, YZ_PLANE } from "../../geom-3d";
import {
  AnyPlaneNode,
  BasePlaneNode,
  PivotedPlaneNode,
  PlaneFromPoints,
  RotatedPlaneNode,
  TranslatedPlaneNode,
} from "../../sketch-nodes";
import { Handler } from "../../sketch-nodes/types";
import { guardAngle, guardPlane, guardPoint3, guardVec3 } from "../guards";
import { getAxis } from "./utils/getAxis";

export interface PlaneHandler<T extends AnyPlaneNode> extends Handler<T> {
  category: "Plane";
  eval: (plane: T, children: unknown[]) => Plane;
}

const basePlaneHandler: PlaneHandler<BasePlaneNode> = {
  category: "Plane",
  nodeType: "BasePlane",
  children: () => [],
  eval: function evalBasePlane(node) {
    if (node.plane === "xy") {
      return XY_PLANE;
    }
    if (node.plane === "yz") {
      return YZ_PLANE;
    }
    if (node.plane === "xz") {
      return XZ_PLANE;
    }
    throw new Error(`Unknown base plane: ${node.plane}`);
  },
};

const TranslatedPlaneHandler: PlaneHandler<TranslatedPlaneNode> = {
  category: "Plane",
  nodeType: "TranslatedPlane",
  children: (node) => [node.vector],
  eval: function evalTranslatedPlane(node, [vector]) {
    return guardPlane(node.plane).translate(guardVec3(vector));
  },
};

const PivotPlaneHandler: PlaneHandler<PivotedPlaneNode> = {
  category: "Plane",
  nodeType: "PivotedPlane",
  children: (node) => {
    if (node.axis === "x" || node.axis === "y" || node.axis === "z") {
      return [node.angle];
    }
    return [node.angle, node.axis];
  },
  eval: function evalPivotPlane(node, [angle, inputAxis]) {
    return guardPlane(node.plane).pivot(
      guardAngle(angle),
      getAxis(node.axis, inputAxis),
    );
  },
};

const RotatedPlaneHandler: PlaneHandler<RotatedPlaneNode> = {
  category: "Plane",
  nodeType: "RotatedPlane",
  children: (node) => [node.angle],
  eval: function evalRotatedPlane(node, [angle]) {
    return guardPlane(node.plane).rotateAroundZ(guardAngle(angle));
  },
};

const planeFromPointsHandler: PlaneHandler<PlaneFromPoints> = {
  category: "Plane",
  nodeType: "PlaneFromPoints",
  children: (node) => [node.origin, node.p1, node.p2],
  eval: function evalPlaneFromPoints(_, [originInput, p1Input, p2Input]) {
    const origin = guardPoint3(originInput);
    const p1 = guardPoint3(p1Input);
    const p2 = guardPoint3(p2Input);

    const xAxis = origin.vecTo(p1).normalize();
    const yAxis = origin.vecTo(p2).normalize();
    const zAxis = xAxis.cross(yAxis) as UnitVec3;

    return new Plane(origin, zAxis, xAxis);
  },
};

export const planeHandlers = [
  basePlaneHandler,
  TranslatedPlaneHandler,
  PivotPlaneHandler,
  RotatedPlaneHandler,
  planeFromPointsHandler,
] as const;
