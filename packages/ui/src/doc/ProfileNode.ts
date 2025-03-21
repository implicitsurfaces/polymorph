import { NodeId, AnyNodeData } from "./Node";
import { ShapeNode, ShapeNodeData, ShapeNodeOptions } from "./ShapeNode";
import { Document } from "./Document";

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface ProfileNodeData extends ShapeNodeData {}

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface ProfileNodeOptions extends ShapeNodeOptions {}

/**
 * Represents a 2D shape.
 */
export abstract class ProfileNode extends ShapeNode {
  abstract get data(): ProfileNodeData;

  constructor(doc: Document, id: NodeId) {
    super(doc, id);
  }

  static dataFromOptions(
    doc: Document,
    options: ProfileNodeOptions,
  ): ProfileNodeData {
    return {
      ...ShapeNode.dataFromOptions(doc, options),
    };
  }

  static dataFromAny(d: AnyNodeData): ProfileNodeData {
    return {
      ...ShapeNode.dataFromAny(d),
    };
  }
}
