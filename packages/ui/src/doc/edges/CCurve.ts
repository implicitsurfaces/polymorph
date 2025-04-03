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

export interface CCurveData extends EdgeNodeData {
  readonly controlPointId: NodeId;
}

export interface CCurveOptions extends EdgeNodeOptions {
  readonly controlPoint: Point;
}

export class CCurve extends EdgeNode {
  static readonly defaultName = "C-Curve";

  private _data: CCurveData;

  get data(): CCurveData {
    return this._data;
  }

  setData(data: CCurveData) {
    this._data = data;
  }

  constructor(doc: Document, id: NodeId, data: CCurveData) {
    super(doc, id);
    this._data = data;
  }

  static dataFromOptions(doc: Document, options: CCurveOptions): CCurveData {
    return {
      ...EdgeNode.dataFromOptions(doc, options),
      controlPointId: options.controlPoint.id,
    };
  }

  static dataFromAny(d: AnyNodeData): CCurveData {
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
    // Note: alternatively, for symmetry, we could either have both startPoint
    // and endPoint be anchors (or none of them), but it isn't clear if this
    // makes the user experience better or worse.
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

registerNodeType(CCurve);
