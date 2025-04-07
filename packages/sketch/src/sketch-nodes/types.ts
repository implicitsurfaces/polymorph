import { AnyAngleNode } from "./angle";
import { AnyConstraintNode } from "./constraint";
import { AnyDistanceNode } from "./distance";
import { AnyEdgeNode } from "./edge";
import { AnyExtrusion2DNode } from "./extrusion-2d";
import { AnyPathNode } from "./path";
import { AnyPlaneNode } from "./plane";
import { AnyPointNode } from "./point";
import { AnyPoint3Node } from "./point3";
import { AnyBaseProfileNode } from "./profile";
import { AnyProfileOperationNode } from "./profile-operation";
import { AnySimpleRealValueNode } from "./real-value";
import { AnySolidOperationNode } from "./solid-operation";
import { AnyBasicSolidNode } from "./solid";
import { AnyVectorNode } from "./vector";
import { AnyVector3Node } from "./vector3";

export type AnyProfileNode =
  | AnyBaseProfileNode
  | AnyProfileOperationNode
  | AnyExtrusion2DNode;

export type AnySolidNode = AnyBasicSolidNode | AnySolidOperationNode;

export type AnyRealValueNode =
  | number
  | AnySimpleRealValueNode
  | AnyDistanceNode;

export type AllSketchNode =
  | AnySimpleRealValueNode
  | AnyDistanceNode
  | AnyPathNode
  | AnyAngleNode
  | AnyEdgeNode
  | AnyPointNode
  | AnyPoint3Node
  | AnyVectorNode
  | AnyVector3Node
  | AnyProfileNode
  | AnySolidNode
  | AnyProfileOperationNode
  | AnyPlaneNode
  | AnyConstraintNode;

export type AnySketchNode = number | AllSketchNode;

export type NodeType = AllSketchNode["nodeType"];
export type NodeCategory = AllSketchNode["category"];

export type CategoryOf<T extends AllSketchNode> = T["category"];

export type NodeCategoryMap = {
  [H in AllSketchNode as H["category"]]: H;
};

export type Handler<T extends AllSketchNode> = {
  category: T["category"];
  nodeType: T["nodeType"];
  children: (node: T) => unknown[];
  eval: (node: T, children: unknown[]) => unknown;
};

export type NodeTypeOf<T extends { nodeType: string }> = T["nodeType"];
