import {
  EdgeNode,
  EdgeNodeData,
  EdgeNodeOptions,
  ControlPoint,
} from "../EdgeNode";

import { NodeId, AnyNodeData, registerNodeType } from "../Node";
import { Document } from "../Document";
import { Point } from "../Point";

import { asNodeId } from "../dataFromAny";

export interface ArcFromStartTangentData extends EdgeNodeData {
  readonly controlPointId: NodeId;
}

export interface ArcFromStartTangentOptions extends EdgeNodeOptions {
  readonly controlPoint: Point;
}

export class ArcFromStartTangent extends EdgeNode {
  static readonly defaultName = "Arc";

  private _data: ArcFromStartTangentData;

  get data(): ArcFromStartTangentData {
    return this._data;
  }

  setData(data: ArcFromStartTangentData) {
    this._data = data;
  }

  constructor(doc: Document, id: NodeId, data: ArcFromStartTangentData) {
    super(doc, id);
    this._data = data;
  }

  static dataFromOptions(
    doc: Document,
    options: ArcFromStartTangentOptions,
  ): ArcFromStartTangentData {
    return {
      ...EdgeNode.dataFromOptions(doc, options),
      controlPointId: options.controlPoint.id,
    };
  }

  static dataFromAny(d: AnyNodeData): ArcFromStartTangentData {
    return {
      ...EdgeNode.dataFromAny(d),
      controlPointId: asNodeId(d, "controlPointId"),
    };
  }

  controlPoints(): ControlPoint[] {
    return [
      {
        edge: this,
        name: "controlPoint",
        prettyName: "Control Point",
        point: this.controlPoint,
        anchor: this.startPoint,
      },
    ];
  }

  get controlPoint(): Point {
    return this.getNodeAs(this.data.controlPointId, Point);
  }

  set controlPoint(point: Point) {
    this._data = {
      ...this.data,
      controlPointId: point.id,
    };
  }
}

registerNodeType(ArcFromStartTangent);
