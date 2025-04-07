import {
  ConeNode,
  ConeSurfaceNode,
  SphereNode,
  ExtrusionNode,
  SolidRotationNode,
} from "../../sketch-nodes";
import { AnySolidNode, Handler } from "../../sketch-nodes/types";
import { SolidDistField } from "../../types";
import { guardAngle, guardNum, guardProfile, guardSolid } from "../guards";
import {
  Sphere as SphereSolid,
  Cone as ConeSolid,
  ConeSurface as ConeSurfaceSolid,
  Extrusion as ExtrusionSolid,
  SolidRotation,
} from "../../solids";
import { XY_PLANE } from "../../geom-3d";
import { getAxis } from "./utils/getAxis";

export interface SolidNodeHandler<T extends AnySolidNode> extends Handler<T> {
  category: "Solid";
  eval: (solid: T, children: unknown[]) => SolidDistField;
}

const sphereHandler: SolidNodeHandler<SphereNode> = {
  category: "Solid",
  nodeType: "Sphere",
  children: (node) => [node.radius],
  eval: function evalSphere(_, [radius]) {
    return new SphereSolid(guardNum(radius));
  },
};

const coneHandler: SolidNodeHandler<ConeNode> = {
  category: "Solid",
  nodeType: "Cone",
  children: (node) => [node.radius, node.height],
  eval: function evalCone(_, [radius, height]) {
    return new ConeSolid(guardNum(radius), guardNum(height));
  },
};

const coneSurfaceHandler: SolidNodeHandler<ConeSurfaceNode> = {
  category: "Solid",
  nodeType: "ConeSurface",
  children: (node) => [node.radius, node.height],
  eval: function evalConeSurface(_, [radius, height]) {
    return new ConeSurfaceSolid(guardNum(radius), guardNum(height));
  },
};

const extrusionHandler: SolidNodeHandler<ExtrusionNode> = {
  category: "Solid",
  nodeType: "Extrusion",
  children: (node) => [node.profile, node.height],
  eval: function evalExtrusion(_, [profile, height]) {
    return new ExtrusionSolid(
      guardNum(height),
      guardProfile(profile),
      XY_PLANE,
    );
  },
};

const solidRotationHandler: SolidNodeHandler<SolidRotationNode> = {
  category: "Solid",
  nodeType: "SolidRotation",
  children: (node) => {
    if (node.axis === "x" || node.axis === "y" || node.axis === "z") {
      return [node.solid, node.angle];
    }
    return [node.solid, node.angle, node.axis];
  },
  eval: function evalSolidRotation(node, [solid, angle, inputAxis]) {
    return new SolidRotation(
      guardAngle(angle),
      guardSolid(solid),
      getAxis(node.axis, inputAxis),
    );
  },
};

export const solidHandlers = [
  sphereHandler,
  coneHandler,
  coneSurfaceHandler,
  extrusionHandler,
  solidRotationHandler,
] as const;
