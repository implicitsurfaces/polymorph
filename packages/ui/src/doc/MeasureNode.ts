import { Node, NodeId, NodeData, NodeOptions, AnyNodeData } from "./Node";
import { Document } from "./Document";
import { asBoolean } from "./dataFromAny";

export interface MeasureNodeData extends NodeData {
  readonly isLocked: boolean;
}

export interface MeasureNodeOptions extends NodeOptions {
  readonly isLocked?: boolean;
}

export abstract class MeasureNode extends Node {
  abstract get data(): MeasureNodeData;

  constructor(doc: Document, id: NodeId) {
    super(doc, id);
  }

  static dataFromOptions(
    doc: Document,
    options: MeasureNodeOptions,
  ): MeasureNodeData {
    return {
      ...Node.dataFromOptions(doc, options),
      isLocked: options.isLocked ?? true,
    };
    // For now, we set the measure to be locked by default.
    // TODO: should the default be unlocked?
  }

  static dataFromAny(d: AnyNodeData): MeasureNodeData {
    return {
      ...Node.dataFromAny(d),
      isLocked: asBoolean(d, "isLocked"),
    };
  }

  abstract updateMeasure(): void;

  get isLocked(): boolean {
    return this.data.isLocked;
  }

  set isLocked(locked: boolean) {
    this.setData({
      ...this.data,
      isLocked: locked,
    });
  }
}
