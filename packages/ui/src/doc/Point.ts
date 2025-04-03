import { Vector2 } from "threejs-math";

import {
  SkeletonNode,
  SkeletonNodeData,
  SkeletonNodeOptions,
} from "./SkeletonNode";

import { NodeId, AnyNodeData, registerNodeType } from "./Node";
import { Document } from "./Document";
import { Number } from "./Number";

import { asNodeId } from "./dataFromAny";
import { getOrCreateNodeId } from "./dataFromOptions";

export interface PointData extends SkeletonNodeData {
  readonly xId: NodeId;
  readonly yId: NodeId;
}

export interface PointOptions extends SkeletonNodeOptions {
  readonly position?: Vector2 | [number, number];
  readonly x?: Number;
  readonly y?: Number;
}

function getVec2Pair(position?: Vector2 | [number, number]): [number, number] {
  if (!position) {
    return [0, 0];
  }
  if (position instanceof Vector2) {
    return [position.x, position.y];
  }
  return position;
}

export class Point extends SkeletonNode {
  static readonly defaultName = "Point";

  private _data: PointData;

  get data(): PointData {
    return this._data;
  }

  setData(data: PointData) {
    this._data = data;
  }

  constructor(doc: Document, id: NodeId, data: PointData) {
    super(doc, id);
    this._data = data;
  }

  static dataFromOptions(doc: Document, options: PointOptions): PointData {
    const defaultPos = getVec2Pair(options.position);
    const xId = getOrCreateNodeId(doc, options.x, Number, {
      layer: options.layer,
      value: defaultPos[0],
    });
    const yId = getOrCreateNodeId(doc, options.y, Number, {
      layer: options.layer,
      value: defaultPos[1],
    });
    return {
      ...SkeletonNode.dataFromOptions(doc, options),
      xId: xId,
      yId: yId,
    };
  }

  static dataFromAny(d: AnyNodeData): PointData {
    return {
      ...SkeletonNode.dataFromAny(d),
      xId: asNodeId(d, "xId"),
      yId: asNodeId(d, "yId"),
    };
  }

  get x(): Number {
    return this.getNodeAs(this.data.xId, Number);
  }

  set x(number: Number) {
    this._data = {
      ...this.data,
      xId: number.id,
    };
  }

  get y(): Number {
    return this.getNodeAs(this.data.yId, Number);
  }

  set y(number: Number) {
    this._data = {
      ...this.data,
      yId: number.id,
    };
  }

  get position(): Vector2 {
    return new Vector2(this.x.value, this.y.value);
  }

  set position(position: Vector2) {
    this.x.value = position.x;
    this.y.value = position.y;
  }
}

registerNodeType(Point);
