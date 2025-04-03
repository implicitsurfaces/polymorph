import {
  EdgeNode,
  EdgeNodeData,
  EdgeNodeOptions,
  ControlPoint,
} from "../EdgeNode";

import { NodeId, AnyNodeData, registerNodeType } from "../Node";
import { Document } from "../Document";

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface LineSegmentData extends EdgeNodeData {}

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface LineSegmentOptions extends EdgeNodeOptions {}

export class LineSegment extends EdgeNode {
  static readonly defaultName = "Line Segment";

  private _data: LineSegmentData;

  get data(): LineSegmentData {
    return this._data;
  }

  setData(data: LineSegmentData) {
    this._data = data;
  }

  constructor(doc: Document, id: NodeId, data: LineSegmentData) {
    super(doc, id);
    this._data = data;
  }

  static dataFromOptions(
    doc: Document,
    options: LineSegmentOptions,
  ): LineSegmentData {
    return {
      ...EdgeNode.dataFromOptions(doc, options),
    };
  }

  static dataFromAny(d: AnyNodeData): LineSegmentData {
    return {
      ...EdgeNode.dataFromAny(d),
    };
  }

  controlPoints(): ControlPoint[] {
    return [];
  }
}

registerNodeType(LineSegment);
