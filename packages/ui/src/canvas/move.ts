import { Vector2 } from "threejs-math";

import { DocumentManager } from "../DocumentManager.ts";
import {
  Document,
  ElementId,
  Point,
  EdgeElement,
  Layer,
  LineSegment,
  ArcFromStartTangent,
  CCurve,
  SCurve,
} from "../Document.ts";
import { Selectable, Selection } from "../Selection.ts";

type OnMoveCallback = (delta: Vector2) => void;

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
interface ControlPoint {
  readonly edge: EdgeElement;
  readonly name: string;
  readonly position: Vector2;
  readonly anchor?: ElementId;
}

export function getControlPoints(edge: EdgeElement): Array<ControlPoint> {
  if (edge instanceof LineSegment) {
    return [];
  } else if (edge instanceof ArcFromStartTangent) {
    return [
      {
        edge: edge,
        name: "controlPoint",
        position: edge.controlPoint,
        anchor: edge.startPoint,
      },
    ];
  } else if (edge instanceof CCurve) {
    return [
      {
        edge: edge,
        name: "controlPoint",
        position: edge.controlPoint,
        anchor: edge.startPoint,
        // Note: alternatively, for symmetry, we could either have both
        // startPoint and endPoint be anchors (or none of them), but it
        // isn't clear if this makes the user experience better or worse.
      },
    ];
  } else if (edge instanceof SCurve) {
    return [
      {
        edge: edge,
        name: "startControlPoint",
        position: edge.startControlPoint,
        anchor: edge.startPoint,
      },
      {
        edge: edge,
        name: "endControlPoint",
        position: edge.endControlPoint,
        anchor: edge.endPoint,
      },
    ];
  }
  return [];
}

// Computes the set of all points that are either selected,
// or that are the endpoint of a selected edge.
//
function computeMovedPoints(
  doc: Document,
  selection: Selection,
): Set<ElementId> {
  const movedPoints = new Set<ElementId>();
  for (const selectable of selection.selected()) {
    if (selectable.type === "Element") {
      const element = doc.getElement(selectable.id);
      if (element instanceof Point) {
        movedPoints.add(element.id);
      } else if (element instanceof EdgeElement) {
        movedPoints.add(element.startPoint);
        movedPoints.add(element.endPoint);
      }
    }
  }
  return movedPoints;
}

// Computes the set of edges that are incident to the given `points`.
//
function computeIncidentEdges(
  doc: Document,
  points: Set<ElementId>,
): Set<EdgeElement> {
  const edges = new Set<EdgeElement>();
  for (const layerId of doc.layers) {
    const layer = doc.getElement(layerId, Layer);
    if (!layer) {
      continue;
    }
    for (const elementId of layer.elements) {
      const edge = doc.getElement(elementId, EdgeElement);
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

// Computes the set of all control points that are explicitly or implicitly
// moved if the given `selection` is to be moved.
//
function computeMovedControlPoints(
  doc: Document,
  selection: Selection,
  movedPoints: Set<ElementId>,
): Set<ControlPoint> {
  const movedControlPoints = new Set<ControlPoint>();

  // Control points that are explicitly selected.
  //
  for (const selectable of selection.selected()) {
    if (selectable.type === "SubElement") {
      const edge = doc.getElement(selectable.id, EdgeElement);
      if (edge) {
        for (const cp of getControlPoints(edge)) {
          if (cp.name === selectable.subName) {
            movedControlPoints.add(cp);
          }
        }
      }
    }
  }

  // Control points that are anchored to moved points.
  //
  const incidentEdges = computeIncidentEdges(doc, movedPoints);
  for (const edge of incidentEdges) {
    for (const cp of getControlPoints(edge)) {
      if (cp.anchor && movedPoints.has(cp.anchor)) {
        movedControlPoints.add(cp);
      }
    }
  }

  return movedControlPoints;
}

// Returns the onMove callback for a Point element.
//
function onPointMove(point: Point): OnMoveCallback {
  const position = point.position.clone();
  return (delta: Vector2) => {
    point.position = position.clone().add(delta);
  };
}

// Returns the onMove callback for a ControlPoint sub-element.
//
function onControlPointMove(cp: ControlPoint): OnMoveCallback | undefined {
  const edge = cp.edge;
  const name = cp.name;
  if (edge instanceof LineSegment) {
    // Nothing
  } else if (edge instanceof ArcFromStartTangent) {
    if (name === "controlPoint") {
      const controlPoint = edge.controlPoint.clone();
      return (delta: Vector2) => {
        edge.controlPoint = controlPoint.clone().add(delta);
      };
    }
  } else if (edge instanceof CCurve) {
    if (name === "controlPoint") {
      const controlPoint = edge.controlPoint.clone();
      return (delta: Vector2) => {
        edge.controlPoint = controlPoint.clone().add(delta);
      };
    }
  } else if (edge instanceof SCurve) {
    if (name === "startControlPoint") {
      const startControlPoint = edge.startControlPoint.clone();
      return (delta: Vector2) => {
        edge.startControlPoint = startControlPoint.clone().add(delta);
      };
    } else if (name === "endControlPoint") {
      const endControlPoint = edge.endControlPoint.clone();
      return (delta: Vector2) => {
        edge.endControlPoint = endControlPoint.clone().add(delta);
      };
    }
  }
  return undefined;
}

// Whether a given selectable is movable or has movable sub-elements
//
function isMovable(doc: Document, selectable: Selectable | undefined): boolean {
  if (!selectable) {
    return false;
  }
  if (selectable.type === "Element") {
    const element = doc.getElement(selectable.id);
    if (!element) {
      return false;
    }
    return element instanceof Point || element instanceof EdgeElement;
  } else if (selectable.type === "SubElement") {
    // For now, all sub-elements are movable, since the only implemented
    // sub-elements are control points which are all movable.
    return true;
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
  const movedControlPoints = computeMovedControlPoints(
    doc,
    selection,
    movedPoints,
  );

  // Store their onMove() callbacks.
  //
  data.onMoves = [];
  for (const id of movedPoints) {
    const point = doc.getElement(id, Point);
    if (point) {
      data.onMoves.push(onPointMove(point));
    }
  }
  for (const cp of movedControlPoints) {
    const onMove = onControlPointMove(cp);
    if (onMove) {
      data.onMoves.push(onMove);
    }
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
