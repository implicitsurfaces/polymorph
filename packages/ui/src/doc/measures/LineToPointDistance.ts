import { NodeId, AnyNodeData, registerNodeType } from "../Node";

import {
  MeasureNode,
  MeasureNodeData,
  MeasureNodeOptions,
} from "../MeasureNode";

import { Document } from "../Document";
import { Number } from "../Number";
import { Point } from "../Point";
import { LineSegment } from "../edges/LineSegment";

import { asNodeId } from "../dataFromAny";
import { getOrCreateNodeId } from "../dataFromOptions";

export interface LineToPointDistanceData extends MeasureNodeData {
  readonly lineId: NodeId;
  readonly pointId: NodeId;
  readonly numberId: NodeId;
}

export interface LineToPointDistanceOptions extends MeasureNodeOptions {
  readonly line: LineSegment;
  readonly point: Point;
  readonly number?: Number;
}

export class LineToPointDistance extends MeasureNode {
  static readonly defaultName = "Line to Point Distance";

  private _data: LineToPointDistanceData;

  get data(): LineToPointDistanceData {
    return this._data;
  }

  setData(data: LineToPointDistanceData) {
    this._data = data;
  }

  constructor(doc: Document, id: NodeId, data: LineToPointDistanceData) {
    super(doc, id);
    this._data = data;
  }

  static dataFromOptions(
    doc: Document,
    options: LineToPointDistanceOptions,
  ): LineToPointDistanceData {
    const numberId = getOrCreateNodeId(doc, options.number, Number, {
      layer: options.layer,
      value: 0,
    });
    return {
      ...MeasureNode.dataFromOptions(doc, options),
      lineId: options.line.id,
      pointId: options.point.id,
      numberId: numberId,
    };
  }

  static dataFromAny(d: AnyNodeData): LineToPointDistanceData {
    return {
      ...MeasureNode.dataFromAny(d),
      lineId: asNodeId(d, "lineId"),
      pointId: asNodeId(d, "pointId"),
      numberId: asNodeId(d, "numberId"),
    };
  }

  get line(): LineSegment {
    return this.getNodeAs(this.data.lineId, LineSegment);
  }

  set line(line: LineSegment) {
    this._data = {
      ...this.data,
      lineId: line.id,
    };
  }

  get point(): Point {
    return this.getNodeAs(this.data.pointId, Point);
  }

  set point(point: Point) {
    this._data = {
      ...this.data,
      startPointId: point.id,
    };
  }

  get number(): Number {
    return this.getNodeAs(this.data.numberId, Number);
  }

  set number(number: Number) {
    this._data = {
      ...this.data,
      numberId: number.id,
    };
  }

  updateMeasure() {
    const p1x = this.line.startPoint.position.x;
    const p1y = this.line.startPoint.position.y;
    const p2x = this.line.endPoint.position.x;
    const p2y = this.line.endPoint.position.y;
    const px = this.point.position.x;
    const py = this.point.position.y;

    // Calculate distance from point to infinite line
    // using formula |ax + by + c| / sqrt(a^2 + b^2)
    // where ax + by + c = 0 is line equation
    const a = p2y - p1y;
    const b = p1x - p2x;
    const c = p2x * p1y - p1x * p2y;

    const distance = Math.abs(a * px + b * py + c) / Math.sqrt(a * a + b * b);

    this.number.value = distance;
  }
}

registerNodeType(LineToPointDistance);
