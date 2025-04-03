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

export interface SCurveData extends EdgeNodeData {
  readonly startControlPointId: NodeId;
  readonly endControlPointId: NodeId;
}

export interface SCurveOptions extends EdgeNodeOptions {
  readonly startControlPoint: Point;
  readonly endControlPoint: Point;
}

export class SCurve extends EdgeNode {
  static readonly defaultName = "S-Curve";

  private _data: SCurveData;

  get data(): SCurveData {
    return this._data;
  }

  setData(data: SCurveData) {
    this._data = data;
  }

  constructor(doc: Document, id: NodeId, data: SCurveData) {
    super(doc, id);
    this._data = data;
  }

  static dataFromOptions(doc: Document, options: SCurveOptions): SCurveData {
    return {
      ...EdgeNode.dataFromOptions(doc, options),
      startControlPointId: options.startControlPoint.id,
      endControlPointId: options.endControlPoint.id,
    };
  }

  static dataFromAny(d: AnyNodeData): SCurveData {
    return {
      ...EdgeNode.dataFromAny(d),
      startControlPointId: asNodeId(d, "startControlPointId"),
      endControlPointId: asNodeId(d, "endControlPointId"),
    };
  }

  controlPoints(): ControlPoint[] {
    return [
      {
        edge: this,
        name: "startControlPoint",
        prettyName: "Start Control Point",
        point: this.startControlPoint,
        anchor: this.startPoint,
      },
      {
        edge: this,
        name: "endControlPoint",
        prettyName: "End Control Point",
        point: this.endControlPoint,
        anchor: this.endPoint,
      },
    ];
  }

  get startControlPoint(): Point {
    return this.getNodeAs(this.data.startControlPointId, Point);
  }

  set startControlPoint(point: Point) {
    this._data = {
      ...this.data,
      startControlPointId: point.id,
    };
  }

  get endControlPoint(): Point {
    return this.getNodeAs(this.data.endControlPointId, Point);
  }

  set endControlPoint(point: Point) {
    this._data = {
      ...this.data,
      endControlPointId: point.id,
    };
  }
}

registerNodeType(SCurve);
