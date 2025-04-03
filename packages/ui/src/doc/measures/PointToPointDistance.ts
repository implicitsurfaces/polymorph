import { NodeId, AnyNodeData, registerNodeType } from "../Node";

import {
  MeasureNode,
  MeasureNodeData,
  MeasureNodeOptions,
} from "../MeasureNode";

import { Document } from "../Document";
import { Number } from "../Number";
import { Point } from "../Point";

import { asNodeId } from "../dataFromAny";
import { getOrCreateNodeId } from "../dataFromOptions";

export interface PointToPointDistanceData extends MeasureNodeData {
  readonly startPointId: NodeId;
  readonly endPointId: NodeId;
  readonly numberId: NodeId;
}

export interface PointToPointDistanceOptions extends MeasureNodeOptions {
  readonly startPoint: Point;
  readonly endPoint: Point;
  readonly number?: Number;
}

export class PointToPointDistance extends MeasureNode {
  static readonly defaultName = "Point to Point Distance";

  private _data: PointToPointDistanceData;

  get data(): PointToPointDistanceData {
    return this._data;
  }

  setData(data: PointToPointDistanceData) {
    this._data = data;
  }

  constructor(doc: Document, id: NodeId, data: PointToPointDistanceData) {
    super(doc, id);
    this._data = data;
  }

  static dataFromOptions(
    doc: Document,
    options: PointToPointDistanceOptions,
  ): PointToPointDistanceData {
    const numberId = getOrCreateNodeId(doc, options.number, Number, {
      layer: options.layer,
      value: 0,
    });
    return {
      ...MeasureNode.dataFromOptions(doc, options),
      startPointId: options.startPoint.id,
      endPointId: options.endPoint.id,
      numberId: numberId,
    };
  }

  static dataFromAny(d: AnyNodeData): PointToPointDistanceData {
    return {
      ...MeasureNode.dataFromAny(d),
      startPointId: asNodeId(d, "startPointId"),
      endPointId: asNodeId(d, "endPointId"),
      numberId: asNodeId(d, "numberId"),
    };
  }

  get startPoint(): Point {
    return this.getNodeAs(this.data.startPointId, Point);
  }

  set startPoint(point: Point) {
    this._data = {
      ...this.data,
      startPointId: point.id,
    };
  }

  get endPoint(): Point {
    return this.getNodeAs(this.data.endPointId, Point);
  }

  set endPoint(point: Point) {
    this._data = {
      ...this.data,
      endPointId: point.id,
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
    const startPosition = this.startPoint.position;
    const endPosition = this.endPoint.position;
    this.number.value = startPosition.distanceTo(endPosition);
  }
}

registerNodeType(PointToPointDistance);
