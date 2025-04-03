import { Node, NodeId, NodeData, NodeOptions, AnyNodeData } from "./Node";
import { Document } from "./Document";

export type SkeletonRole = "shape" | "construction";

function isSkeletonRole(v: unknown): v is SkeletonRole {
  return v === "shape" || v === "construction";
}

export interface SkeletonNodeData extends NodeData {
  readonly role: SkeletonRole;
}

export interface SkeletonNodeOptions extends NodeOptions {
  readonly role?: SkeletonRole;
}

export abstract class SkeletonNode extends Node {
  abstract get data(): SkeletonNodeData;

  constructor(doc: Document, id: NodeId) {
    super(doc, id);
  }

  static dataFromOptions(
    doc: Document,
    options: SkeletonNodeOptions,
  ): SkeletonNodeData {
    return {
      ...Node.dataFromOptions(doc, options),
      role: options.role ?? "shape",
    };
  }

  static dataFromAny(d: AnyNodeData): SkeletonNodeData {
    return {
      ...Node.dataFromAny(d),
      role: isSkeletonRole(d.role) ? d.role : "shape",
    };
  }

  get role(): SkeletonRole {
    return this.data.role;
  }
}
