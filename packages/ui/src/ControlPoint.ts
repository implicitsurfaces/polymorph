import {
  Document,
  NodeId,
  Point,
  EdgeNode,
  LineSegment,
  ArcFromStartTangent,
  CCurve,
  SCurve,
} from "./Document";

// Stores information about a control point.
//
// If provided, the `anchor` represents a point such that moving the anchor
// should also move the control point, as a convenience to the user.
//
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
  readonly anchor?: NodeId;
}

export function getControlPoints(
  doc: Document,
  edge: EdgeNode,
): Array<ControlPoint> {
  const res: Array<ControlPoint> = [];
  if (edge instanceof LineSegment) {
    // No control points
  } else if (edge instanceof ArcFromStartTangent) {
    const controlPoint = doc.getNode(edge.controlPoint, Point);
    if (controlPoint) {
      res.push({
        edge: edge,
        name: "controlPoint",
        prettyName: "Control Point",
        point: controlPoint,
        anchor: edge.startPoint,
      });
    }
  } else if (edge instanceof CCurve) {
    const controlPoint = doc.getNode(edge.controlPoint, Point);
    if (controlPoint) {
      res.push({
        edge: edge,
        name: "controlPoint",
        prettyName: "Control Point",
        point: controlPoint,
        anchor: edge.startPoint,
        // Note: alternatively, for symmetry, we could either have both
        // startPoint and endPoint be anchors (or none of them), but it
        // isn't clear if this makes the user experience better or worse.
      });
    }
  } else if (edge instanceof SCurve) {
    const startControlPoint = doc.getNode(edge.startControlPoint, Point);
    if (startControlPoint) {
      res.push({
        edge: edge,
        name: "startControlPoint",
        prettyName: "Start Control Point",
        point: startControlPoint,
        anchor: edge.startPoint,
      });
    }
    const endControlPoint = doc.getNode(edge.endControlPoint, Point);
    if (endControlPoint) {
      res.push({
        edge: edge,
        name: "endControlPoint",
        prettyName: "End Control Point",
        point: endControlPoint,
        anchor: edge.endPoint,
      });
    }
  }
  return res;
}
