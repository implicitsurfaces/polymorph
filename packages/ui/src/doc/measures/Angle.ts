import { NodeId, AnyNodeData, registerNodeType } from "../Node";

import {
  MeasureNode,
  MeasureNodeData,
  MeasureNodeOptions,
} from "../MeasureNode";

import { Document } from "../Document";
import { Number } from "../Number";
import { LineSegment } from "../edges/LineSegment";

import { asNodeId } from "../dataFromAny";
import { getOrCreateNodeId } from "../dataFromOptions";

export interface AngleData extends MeasureNodeData {
  readonly line0Id: NodeId;
  readonly line1Id: NodeId;
  readonly numberId: NodeId;
}

export interface AngleOptions extends MeasureNodeOptions {
  readonly line0: LineSegment;
  readonly line1: LineSegment;
  readonly number?: Number;
}

export class Angle extends MeasureNode {
  static readonly defaultName = "Line to Line Angle";

  private _data: AngleData;

  get data(): AngleData {
    return this._data;
  }

  setData(data: AngleData) {
    this._data = data;
  }

  constructor(doc: Document, id: NodeId, data: AngleData) {
    super(doc, id);
    this._data = data;
  }

  static dataFromOptions(doc: Document, options: AngleOptions): AngleData {
    const numberId = getOrCreateNodeId(doc, options.number, Number, {
      layer: options.layer,
      value: 0,
    });
    return {
      ...MeasureNode.dataFromOptions(doc, options),
      line0Id: options.line0.id,
      line1Id: options.line1.id,
      numberId: numberId,
    };
  }

  static dataFromAny(d: AnyNodeData): AngleData {
    return {
      ...MeasureNode.dataFromAny(d),
      line0Id: asNodeId(d, "line0Id"),
      line1Id: asNodeId(d, "line1Id"),
      numberId: asNodeId(d, "numberId"),
    };
  }

  get line0(): LineSegment {
    return this.getNodeAs(this.data.line0Id, LineSegment);
  }

  set line0(line: LineSegment) {
    this._data = {
      ...this.data,
      line0Id: line.id,
    };
  }

  get line1(): LineSegment {
    return this.getNodeAs(this.data.line1Id, LineSegment);
  }

  set line1(line: LineSegment) {
    this._data = {
      ...this.data,
      line1Id: line.id,
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
    const p1x = this.line0.startPoint.position.x;
    const p1y = this.line0.startPoint.position.y;
    const p2x = this.line0.endPoint.position.x;
    const p2y = this.line0.endPoint.position.y;
    const p3x = this.line1.startPoint.position.x;
    const p3y = this.line1.startPoint.position.y;
    const p4x = this.line1.endPoint.position.x;
    const p4y = this.line1.endPoint.position.y;

    // Calculate vectors for both lines
    const v1x = p2x - p1x;
    const v1y = p2y - p1y;
    const v2x = p4x - p3x;
    const v2y = p4y - p3y;

    // Calculate angle between vectors using dot product
    const dot = v1x * v2x + v1y * v2y;
    const mag1 = Math.sqrt(v1x * v1x + v1y * v1y);
    const mag2 = Math.sqrt(v2x * v2x + v2y * v2y);

    let angle = Math.acos(dot / (mag1 * mag2));

    // Convert to degrees
    angle = (angle * 180) / Math.PI;

    // Normalize to 0-180 range
    angle = angle > 180 ? angle - 180 : angle;

    this.number.value = angle;
  }
}

registerNodeType(Angle);
