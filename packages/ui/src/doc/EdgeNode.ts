import {
  SkeletonNode,
  SkeletonNodeData,
  SkeletonNodeOptions,
} from "./SkeletonNode";

import { NodeId, AnyNodeData } from "./Node";
import { Document } from "./Document";
import { Point } from "./Point";

import { asNodeId } from "./dataFromAny";

/**
 * Stores information about a control point.
 *
 * If provided, the `anchor` represents a point such that moving the anchor
 * should also move the control point, as a convenience to the user.
 */
// Notes:
//
// 1. We considered having control points be stored as a position relative to
// their anchor, instead of an absolute position, but we decided that this
// would not play well with the constraint solver.
//
// 2. If the control points are stored as a position relative to their anchor,
// then some care must be done to avoid double-moving the control point if
// both the control point and its anchor are explicitly selected.
//
export interface ControlPoint {
  readonly edge: EdgeNode;
  readonly name: string;
  readonly prettyName: string;
  readonly point: Point;
  readonly anchor?: Point;
}

export interface EdgeNodeData extends SkeletonNodeData {
  readonly startPointId: NodeId;
  readonly endPointId: NodeId;
}

export interface EdgeNodeOptions extends SkeletonNodeOptions {
  readonly startPoint: Point;
  readonly endPoint: Point;
}

export abstract class EdgeNode extends SkeletonNode {
  abstract get data(): EdgeNodeData;

  constructor(doc: Document, id: NodeId) {
    super(doc, id);
  }

  static dataFromOptions(
    doc: Document,
    options: EdgeNodeOptions,
  ): EdgeNodeData {
    return {
      ...SkeletonNode.dataFromOptions(doc, options),
      startPointId: options.startPoint.id,
      endPointId: options.endPoint.id,
    };
  }

  static dataFromAny(d: AnyNodeData): EdgeNodeData {
    return {
      ...SkeletonNode.dataFromAny(d),
      startPointId: asNodeId(d, "startPointId"),
      endPointId: asNodeId(d, "endPointId"),
    };
  }

  abstract controlPoints(): ControlPoint[];

  get startPoint(): Point {
    return this.getNodeAs(this.data.startPointId, Point);
  }

  set startPoint(point: Point) {
    this.setData({
      ...this.data,
      startPointId: point.id,
    });
  }

  get endPoint(): Point {
    return this.getNodeAs(this.data.endPointId, Point);
  }

  set endPoint(point: Point) {
    this.setData({
      ...this.data,
      endPointId: point.id,
    });
  }
}
