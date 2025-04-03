import {
  Node,
  NodeId,
  NodeData,
  NodeOptions,
  AnyNodeData,
  registerNodeType,
} from "./Node";

import { Document } from "./Document";

import { asNodeIdArray } from "./dataFromAny";

export interface LayerData extends NodeData {
  readonly nodeIds: readonly NodeId[];
}

export interface LayerOptions extends NodeOptions {
  readonly nodes?: readonly Node[];
}

export class Layer extends Node {
  static readonly defaultName = "Layer";

  private _data: LayerData;

  get data(): LayerData {
    return this._data;
  }

  setData(data: LayerData) {
    this._data = data;
  }

  constructor(doc: Document, id: NodeId, data: LayerData) {
    super(doc, id);
    this._data = data;
  }

  static dataFromOptions(doc: Document, options: LayerOptions): LayerData {
    return {
      ...Node.dataFromOptions(doc, options),
      nodeIds: options.nodes?.map((node) => node.id) ?? [],
    };
  }

  static dataFromAny(d: AnyNodeData): LayerData {
    return {
      ...Node.dataFromAny(d),
      nodeIds: asNodeIdArray(d, "nodeIds"),
    };
  }

  get nodes(): Node[] {
    return this.data.nodeIds.map((id) => this.getNodeAs(id, Node));
  }

  addNode(node: Node) {
    this._data = {
      ...this.data,
      nodeIds: [...this.data.nodeIds, node.id],
    };
  }

  removeNode(node: Node) {
    const newNodeIds: NodeId[] = [];
    for (const id of this.data.nodeIds) {
      if (id !== node.id) {
        newNodeIds.push(node.id);
      }
    }
    this._data = {
      ...this.data,
      nodeIds: newNodeIds,
    };
  }
}

registerNodeType(Layer);
