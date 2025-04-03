import {
  Node,
  NodeId,
  NodeData,
  NodeOptions,
  AnyNodeData,
  registerNodeType,
} from "./Node";

import { Document } from "./Document";

import { asNumber } from "./dataFromAny";

export interface NumberData extends NodeData {
  readonly value: number;
}

export interface NumberOptions extends NodeOptions {
  readonly value?: number;
}

export class Number extends Node {
  static readonly defaultName = "Number";

  private _data: NumberData;

  get data(): NumberData {
    return this._data;
  }

  setData(data: NumberData) {
    this._data = data;
  }

  constructor(doc: Document, id: NodeId, data: NumberData) {
    super(doc, id);
    this._data = data;
  }

  static dataFromOptions(doc: Document, options: NumberOptions): NumberData {
    return {
      ...Node.dataFromOptions(doc, options),
      value: options.value ?? 0,
    };
  }

  static dataFromAny(d: AnyNodeData): NumberData {
    return {
      ...Node.dataFromAny(d),
      value: asNumber(d, "value"),
    };
  }

  get value(): number {
    return this.data.value;
  }

  set value(newValue: number) {
    this._data = {
      ...this.data,
      value: newValue,
    };
  }
}

registerNodeType(Number);
