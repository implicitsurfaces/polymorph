import { Node, NodeId, NodeData, NodeOptions, AnyNodeData } from "./Node";
import { Document } from "./Document";

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface ShapeNodeData extends NodeData {}

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface ShapeNodeOptions extends NodeOptions {}

/**
 * Represents either a 2D shape (= "profile") or a 3D shape (= "solid").
 */
export abstract class ShapeNode extends Node {
  abstract get data(): ShapeNodeData;

  constructor(doc: Document, id: NodeId) {
    super(doc, id);
  }

  static dataFromOptions(
    doc: Document,
    options: ShapeNodeOptions,
  ): ShapeNodeData {
    return {
      ...Node.dataFromOptions(doc, options),
    };
  }

  static dataFromAny(d: AnyNodeData): ShapeNodeData {
    return {
      ...Node.dataFromAny(d),
    };
  }
}
