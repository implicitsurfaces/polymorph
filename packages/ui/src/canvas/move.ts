import { Vector2 } from "threejs-math";

import { DocumentManager } from "../doc/DocumentManager";
import { Document } from "../doc/Document";
import { Point } from "../doc/Point";
import { EdgeNode, ControlPoint } from "../doc/EdgeNode";
import { Layer } from "../doc/Layer";
import { SkeletonNode } from "../doc/SkeletonNode";
import { Selectable, Selection } from "../doc/Selection";

type OnMoveCallback = (delta: Vector2) => void;

// Computes the set of all points that are either selected,
// or that are the endpoint of a selected edge.
//
function computeMovedPoints(doc: Document, selection: Selection): Set<Point> {
  const movedPoints = new Set<Point>();
  for (const node of doc.getNodes(selection.selectedNodeIds(), SkeletonNode)) {
    if (node instanceof Point) {
      movedPoints.add(node);
    } else if (node instanceof EdgeNode) {
      movedPoints.add(node.startPoint);
      movedPoints.add(node.endPoint);
    }
  }
  return movedPoints;
}

// Computes the set of edges that are incident to the given `points`.
//
function computeIncidentEdges(
  doc: Document,
  points: Set<Point>,
): Set<EdgeNode> {
  const edges = new Set<EdgeNode>();
  for (const layerId of doc.layers) {
    const layer = doc.getNode(layerId, Layer);
    if (!layer) {
      continue;
    }
    for (const node of layer.nodes) {
      if (node instanceof EdgeNode) {
        const edge = node;
        if (points.has(edge.startPoint) || points.has(edge.endPoint)) {
          edges.add(edge);
        }
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
  movedPoints: Set<Point>,
): Set<ControlPoint> {
  const movedControlPoints = new Set<ControlPoint>();
  const incidentEdges = computeIncidentEdges(doc, movedPoints);
  for (const edge of incidentEdges) {
    for (const cp of edge.controlPoints()) {
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
  const position = point.position;
  return (delta: Vector2) => {
    point.position = position.clone().add(delta);
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
    public allMovedPoints: Point[] = [], // Points + Control Points
    public onMoves: OnMoveCallback[] = [],
  ) {}

  clear() {
    this.isMoving = false;
    this.allMovedPoints = [];
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
  // If the hovered object is part of a larger selection, then we want to move
  // the whole selection.
  //
  // Otherwise, we want to only move the hovered object and set it as new
  // selection.
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

  // Compute the set of all moved points and store their onMove() callbacks.
  //
  const allMovedPoints = new Set<Point>();
  data.onMoves = [];
  for (const point of movedPoints) {
    data.onMoves.push(onPointMove(point));
    allMovedPoints.add(point);
  }
  for (const cp of movedControlPoints) {
    data.onMoves.push(onControlPointMove(cp));
    allMovedPoints.add(cp.point);
  }
  data.allMovedPoints = Array.from(allMovedPoints);

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
  documentManager.notifyChanges({
    commit: false,
    solveConstraints: {
      movedPoints: data.allMovedPoints,
    },
  });
}

function end(data: MoveData, documentManager: DocumentManager) {
  data.clear();
  documentManager.notifyChanges({ commit: true });
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
