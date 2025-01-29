import { Vector2 } from "threejs-math";

import { DocumentManager } from "../DocumentManager.ts";
import {
  Document,
  ElementId,
  Point,
  EdgeElement,
  isEdgeElement,
} from "../Document.ts";
import { Selectable, Selection } from "../Selection.ts";

// TODO: Refactor these out of `canvas/drawEdges.ts` and move this `move.ts`
// file in a different folder, e.g., the `tools` folder.
//
import { getEdgeShapesAndControls, ControlPoint } from "./drawEdges.ts";

type OnMoveCallback = (delta: Vector2) => void;

// Adds to the given `movedPoints` set all the points IDs that are explicitly
// or implicitly moved if the given `selectable` is to be moved.
//
function populateMovedPoints(
  doc: Document,
  selectable: Selectable,
  movedPoints: Set<ElementId>,
) {
  if (selectable.type === "Element") {
    const element = doc.getElementFromId(selectable.id);
    if (!element) {
      return;
    }
    if (element.type === "Point") {
      movedPoints.add(element.id);
    } else if (isEdgeElement(element)) {
      movedPoints.add(element.startPoint);
      movedPoints.add(element.endPoint);
    }
  }
}

// Computes the set of all points that are explicitly or implicitly moved if
// the given `selection` is to be moved.
//
function computeMovedPoints(
  doc: Document,
  selection: Selection,
): Set<ElementId> {
  const movedPoints = new Set<ElementId>();
  for (const s of selection.selected()) {
    populateMovedPoints(doc, s, movedPoints);
  }
  return movedPoints;
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
function onControlPointMove(
  edge: EdgeElement,
  controlPointName: string,
): OnMoveCallback | undefined {
  switch (edge.type) {
    case "LineSegment": {
      break;
    }
    case "ArcFromStartTangent": {
      if (controlPointName === "tangent") {
        const tangent = edge.tangent.clone();
        return (delta: Vector2) => {
          edge.tangent = tangent.clone().add(delta);
        };
      }
      break;
    }
    case "CCurve": {
      if (controlPointName === "controlPoint") {
        const controlPoint = edge.controlPoint.clone();
        return (delta: Vector2) => {
          edge.controlPoint = controlPoint.clone().add(delta);
        };
      }
      break;
    }
    case "SCurve": {
      if (controlPointName === "startControlPoint") {
        const startControlPoint = edge.startControlPoint.clone();
        return (delta: Vector2) => {
          edge.startControlPoint = startControlPoint.clone().add(delta);
        };
      } else if (controlPointName === "endControlPoint") {
        const endControlPoint = edge.endControlPoint.clone();
        return (delta: Vector2) => {
          edge.endControlPoint = endControlPoint.clone().add(delta);
        };
      }
      break;
    }
  }
  return undefined;
}

function populateControlPointOnMove(
  edge: EdgeElement,
  cp: ControlPoint,
  movedPoints: Set<ElementId>,
  onMoves: Array<OnMoveCallback>,
) {
  if (!cp.relativeTo || !movedPoints.has(cp.relativeTo)) {
    const onMove = onControlPointMove(edge, cp.name);
    if (onMove) {
      onMoves.push(onMove);
    }
  }
}

// Note: if an edge and one its a control points are both selected, then the
// onMove() callback of the control point will be twice in `onMoves`, but
// it's okay since they simply compute the same value from their cached
// onMousePress position and the provided delta (it won't be moved "twice as
// much").
//
function populateControlPointOnMoves(
  doc: Document,
  edge: EdgeElement,
  movedPoints: Set<ElementId>,
  onMoves: Array<OnMoveCallback>,
) {
  const startPoint = doc.getElementFromId<Point>(edge.startPoint);
  const endPoint = doc.getElementFromId<Point>(edge.endPoint);
  if (!startPoint || !endPoint) {
    return;
  }
  // TODO: cache the controls from the draw call?
  const sc = getEdgeShapesAndControls(doc, edge);
  for (const cp of sc.controlPoints) {
    populateControlPointOnMove(edge, cp, movedPoints, onMoves);
  }
}

// Whether a given selectable is movable or has movable sub-elements
//
function isMovable(doc: Document, selectable: Selectable | undefined): boolean {
  if (!selectable) {
    return false;
  }
  if (selectable.type === "Element") {
    const element = doc.getElementFromId(selectable.id);
    if (!element) {
      return false;
    }
    return element.type === "Point" || isEdgeElement(element);
  } else if (selectable.type === "SubElement") {
    // For now, all sub-elements are movable, since the only implemented
    // sub-elements are control points which are all movable.
    return true;
  }
  return false;
}

function populateNonPointOnMoves(
  doc: Document,
  selectable: Selectable,
  movedPoints: Set<ElementId>,
  onMoves: Array<OnMoveCallback>,
) {
  if (selectable.type === "Element") {
    const element = doc.getElementFromId(selectable.id);
    if (element && isEdgeElement(element)) {
      populateControlPointOnMoves(doc, element, movedPoints, onMoves);
    }
  } else if (selectable.type === "SubElement") {
    const element = doc.getElementFromId(selectable.id);
    if (element && isEdgeElement(element)) {
      // TODO: Avoid having to recompute the ControlPoints?
      // For now we need it as we need `cp.relativeTo` which is
      // not stored in the Selectable, but perhaps we could refactor
      // the code differently to make this cleaner/faster.
      const sc = getEdgeShapesAndControls(doc, element);
      for (const cp of sc.controlPoints) {
        if (cp.name === selectable.subName) {
          populateControlPointOnMove(element, cp, movedPoints, onMoves);
        }
      }
    }
  }
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

  // Make a first pass to collect all explicitly or implicitly moved points.
  // We use this to avoid double-moving points or control points that are
  // already implicitly moved via an edge or another point.
  //
  const movedPoints = computeMovedPoints(doc, selection);

  // Add the onMove() callbacks for all moved points.
  //
  data.onMoves = [];
  for (const id of movedPoints) {
    const point = doc.getElementFromId<Point>(id);
    if (point) {
      data.onMoves.push(onPointMove(point));
    }
  }

  // Make a second pass to collect all the onMove() callbacks of non-point
  // objects.
  //
  for (const s of selection.selected()) {
    populateNonPointOnMoves(doc, s, movedPoints, data.onMoves);
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
