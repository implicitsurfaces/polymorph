import { Vector2 } from "threejs-math";

import { DocumentManager } from "../DocumentManager.ts";
import { Document, NodeId, Point, EdgeNode, Layer } from "../Document.ts";
import { ControlPoint, getControlPoints } from "../ControlPoint.ts";
import { Selectable, Selection } from "../Selection.ts";

type OnMoveCallback = (delta: Vector2) => void;

// Computes the set of all points that are either selected,
// or that are the endpoint of a selected edge.
//
function computeMovedPoints(doc: Document, selection: Selection): Set<NodeId> {
  const movedPoints = new Set<NodeId>();
  for (const selectable of selection.selected()) {
    if (selectable.type === "Node") {
      const node = doc.getNode(selectable.id);
      if (node instanceof Point) {
        movedPoints.add(node.id);
      } else if (node instanceof EdgeNode) {
        movedPoints.add(node.startPoint);
        movedPoints.add(node.endPoint);
      }
    }
  }
  return movedPoints;
}

// Computes the set of edges that are incident to the given `points`.
//
function computeIncidentEdges(
  doc: Document,
  points: Set<NodeId>,
): Set<EdgeNode> {
  const edges = new Set<EdgeNode>();
  for (const layerId of doc.layers) {
    const layer = doc.getNode(layerId, Layer);
    if (!layer) {
      continue;
    }
    for (const nodeId of layer.nodes) {
      const edge = doc.getNode(nodeId, EdgeNode);
      if (!edge) {
        continue;
      }
      if (points.has(edge.startPoint) || points.has(edge.endPoint)) {
        edges.add(edge);
      }
    }
  }
  return edges;
}

// Computes the set of all control points that are implicitly
// moved if the given `selection` is to be moved.
//
function computeMovedControlPoints(
  doc: Document,
  movedPoints: Set<NodeId>,
): Set<ControlPoint> {
  const movedControlPoints = new Set<ControlPoint>();
  const incidentEdges = computeIncidentEdges(doc, movedPoints);
  for (const edge of incidentEdges) {
    for (const cp of getControlPoints(doc, edge)) {
      if (cp.anchor && movedPoints.has(cp.anchor)) {
        movedControlPoints.add(cp);
      }
    }
  }
  return movedControlPoints;
}

// Returns the onMove callback for a Point node.
//
function onPointMove(point: Point): OnMoveCallback {
  const position = point.getPosition();
  return (delta: Vector2) => {
    point.setPosition(position.clone().add(delta));
  };
}

// Returns the onMove callback for a ControlPoint sub-node.
//
function onControlPointMove(cp: ControlPoint): OnMoveCallback {
  return onPointMove(cp.point);
}

// Whether a given selectable is movable or has movable sub-nodes
//
function isMovable(doc: Document, selectable: Selectable | undefined): boolean {
  if (!selectable) {
    return false;
  }
  if (selectable.type === "Node") {
    const node = doc.getNode(selectable.id);
    if (!node) {
      return false;
    }
    return node instanceof Point || node instanceof EdgeNode;
  }
  return false;
}

class MoveData {
  constructor(
    public isMoving: boolean = false,
    public onMoves: Array<OnMoveCallback> = [],
  ) {}

  clear() {
    this.isMoving = false;
    this.onMoves = [];
  }
}

function start(data: MoveData, documentManager: DocumentManager): boolean {
  // Check whether the hovered object is movable, otherwise
  // there is nothing to move and we can fast-return.
  //
  const doc = documentManager.document();
  const selection = documentManager.selection();
  const hovered = selection.hovered();
  const hoveredMovable = isMovable(doc, hovered);
  if (!hovered || !hoveredMovable) {
    return false;
  }

  // Compute which objects we should move.
  //
  // If the hovered object is part of a larger selection, then we want to
  // move the whole selection.
  //
  // Otherwise, we want to only
  // move the hovered object and set it as new selection.
  //
  if (!selection.isSelected(hovered)) {
    selection.setSelected([hovered]);
  }

  // Compute which objects to move. Note that an edge implicitly moves its
  // endpoints, and a point implicitly moves all its anchored control points
  // in all their incident edges.
  //
  // Legend:
  // = selected edge
  // - unselected edge
  // O unselected point
  // X unselected control point
  //
  //  moved as endpoint of moved edge
  //             v
  // O===X===X===O---X---X---O
  //                 ^
  //      moved as anchored control point of implicitly moved point
  //
  // This is why we need the two passes below.
  //
  const movedPoints = computeMovedPoints(doc, selection);
  const movedControlPoints = computeMovedControlPoints(doc, movedPoints);

  // Store their onMove() callbacks.
  //
  data.onMoves = [];
  for (const id of movedPoints) {
    const point = doc.getNode(id, Point);
    if (point) {
      data.onMoves.push(onPointMove(point));
    }
  }
  for (const cp of movedControlPoints) {
    data.onMoves.push(onControlPointMove(cp));
  }

  data.isMoving = true;
  return true;
}

function move(
  data: MoveData,
  documentManager: DocumentManager,
  delta: Vector2,
) {
  for (const onMove of data.onMoves) {
    onMove(delta);
  }
  documentManager.stageChanges();
}

function end(data: MoveData, documentManager: DocumentManager) {
  data.clear();
  documentManager.commitChanges();
}

export class Mover {
  private data: MoveData = new MoveData();

  constructor(readonly documentManager: DocumentManager) {}

  start(): boolean {
    return start(this.data, this.documentManager);
  }

  move(delta: Vector2) {
    move(this.data, this.documentManager, delta);
  }

  end() {
    end(this.data, this.documentManager);
  }
}
