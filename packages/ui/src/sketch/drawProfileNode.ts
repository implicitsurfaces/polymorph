import { ProfileNode } from "../doc/ProfileNode";
import { EdgeCycleProfile } from "../doc/profiles/EdgeCycleProfile";
import { Halfedge } from "../doc/profiles/Halfedge";

import { LineSegment } from "../doc/edges/LineSegment";
import { ArcFromStartTangent } from "../doc/edges/ArcFromStartTangent";
import { CCurve } from "../doc/edges/CCurve";
import { SCurve } from "../doc/edges/SCurve";

import { draw, ProfileEditor } from "draw-api";

// Note: in draw-api/vite.config.js, if dts.rollupTypes is true (which is the
// case as of writing this), then PointMaker and EdgeMaker are not marked as
// exported for some reason, and therefore we cannot do:
//
// import { PointMaker, EdgeMaker } from "draw-api"
//
// We use the following `any` aliases as a workaround.

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type EdgeMaker = any;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type PointMaker = any;

function addEdge(d: EdgeMaker, halfedge: Halfedge): PointMaker {
  const edge = halfedge.edge;
  if (edge instanceof LineSegment) {
    return d.line();
  }
  if (edge instanceof ArcFromStartTangent) {
    if (halfedge.direction) {
      return d.arcFromStartControl(edge.controlPoint.position);
    } else {
      return d.arcFromEndControl(edge.controlPoint.position);
    }
  }
  if (edge instanceof CCurve) {
    return d.CCurve(edge.controlPoint.position);
  }
  if (edge instanceof SCurve) {
    if (halfedge.direction) {
      return d.SCurve(
        edge.startControlPoint.position,
        edge.endControlPoint.position,
      );
    } else {
      return d.SCurve(
        edge.endControlPoint.position,
        edge.startControlPoint.position,
      );
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
  let d = draw(startPoint.position.toArray() as [number, number]);
  for (const [i, halfedge] of halfedges.entries()) {
    const e = addEdge(d, halfedge);
    if (i < lastIndex) {
      d = e.to(halfedge.endPoint().position);
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
