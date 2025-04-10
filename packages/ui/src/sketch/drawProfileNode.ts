import {
  ProfileNode,
  EdgeCycleProfile,
  Halfedge,
  Point,
  LineSegment,
  ArcFromStartTangent,
  CCurve,
  SCurve,
} from "../doc";

import { draw, ProfileEditor, PointMaker, EdgeMaker } from "draw-api";

/**
 * Returns the position of the `Point` as a [number, number] which
 * draw-api needs instead of a Vector2.
 */
function pos(point: Point): [number, number] {
  return [point.x.value, point.y.value];
}

function addEdge(d: EdgeMaker, halfedge: Halfedge): PointMaker {
  const edge = halfedge.edge;
  if (edge instanceof LineSegment) {
    return d.line();
  }
  if (edge instanceof ArcFromStartTangent) {
    if (halfedge.direction) {
      return d.arcFromStartControl(pos(edge.controlPoint));
    } else {
      return d.arcFromEndControl(pos(edge.controlPoint));
    }
  }
  if (edge instanceof CCurve) {
    return d.CCurve(pos(edge.controlPoint));
  }
  if (edge instanceof SCurve) {
    if (halfedge.direction) {
      return d.SCurve(pos(edge.startControlPoint), pos(edge.endControlPoint));
    } else {
      return d.SCurve(pos(edge.endControlPoint), pos(edge.startControlPoint));
    }
  }
  // Fallback to  LineSegment
  return d.line();
}

export function drawEdgeCycleProfile(
  node: EdgeCycleProfile,
): ProfileEditor | undefined {
  const halfedges = node.cycle.halfedges;
  const n = halfedges.length;
  if (n === 0) {
    return undefined;
  }
  const startPoint = halfedges[0].startPoint();
  const lastIndex = n - 1;
  let d = draw(pos(startPoint));
  for (const [i, halfedge] of halfedges.entries()) {
    const e = addEdge(d, halfedge);
    if (i < lastIndex) {
      d = e.to(pos(halfedge.endPoint()));
    } else {
      return e.close();
    }
  }
  return undefined;
}

export function drawProfileNode(node: ProfileNode): ProfileEditor | undefined {
  if (node instanceof EdgeCycleProfile) {
    return drawEdgeCycleProfile(node);
  }
  return undefined;
}
