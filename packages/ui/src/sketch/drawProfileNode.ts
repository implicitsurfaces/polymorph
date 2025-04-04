import { ProfileNode } from "../doc/ProfileNode";
import { EdgeCycleProfile } from "../doc/profiles/EdgeCycleProfile";
import { draw } from "draw-api";

export function drawEdgeCycleProfile(node: EdgeCycleProfile) {
  const halfedges = node.cycle.halfedges;
  const n = halfedges.length;
  if (n === 0) {
    return undefined;
  }
  const startPoint = halfedges[0].startPoint();
  const lastIndex = n - 1;
  let edgeMaker = draw(startPoint.position);
  for (const [i, halfedge] of halfedges.entries()) {
    // TODO: handle all edge types instead of assuming line segment
    const pointMaker = edgeMaker.line();
    if (i < lastIndex) {
      edgeMaker = edgeMaker.line().to(halfedge.endPoint().position);
    } else {
      return pointMaker.close();
    }
  }
  return undefined;
}

export function drawProfileNode(node: ProfileNode) {
  if (node instanceof EdgeCycleProfile) {
    return drawEdgeCycleProfile(node);
  }
  return undefined;
}
