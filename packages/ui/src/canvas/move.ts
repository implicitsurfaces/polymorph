import { Vector2 } from "threejs-math";
import { useMemo } from "react";

import { DocumentManager } from "../DocumentManager.ts";
import { Document, Element, isEdgeElement } from "../Document.ts";
import { Selection, Selectable } from "../Selection.ts";

import { getEdgeShapesAndControls } from "./drawEdges.ts";

interface Movable {
  readonly selectable: Selectable;
  readonly element: Element;
  readonly onMove: (delta: Vector2, selection: Selection) => void;
}

function getMovable(
  doc: Document,
  selectable: Selectable | undefined,
): Movable | undefined {
  if (!selectable) {
    return undefined;
  }
  if (selectable.type === "Element") {
    const element = doc.getElementFromId(selectable.id);
    if (element && element.type === "Point") {
      const position = element.position.clone();
      return {
        selectable: selectable,
        element: element,
        onMove: (delta: Vector2) => {
          element.position = position.clone().add(delta);
        },
      };
    }
  } else if (selectable.type === "SubElement") {
    const element = doc.getElementFromId(selectable.id);
    if (element && isEdgeElement(element)) {
      // TODO: cache the controls from the draw call?
      const sc = getEdgeShapesAndControls(doc, element);
      for (const cp of sc.controlPoints) {
        if (cp.name == selectable.subName) {
          return {
            selectable: selectable,
            element: element,
            onMove: cp.onMove,
          };
        }
      }
    }
  }
  return undefined;
}

class MoveData {
  constructor(
    public isMoving: boolean = false,
    public movables: Array<Movable> = [],
  ) {}
}

function start(data: MoveData, documentManager: DocumentManager): boolean {
  // This is important otherwise in React strict mode moveStart() might be
  // called twice, and the second time the elements might have already moved
  // a little, so we wouldn't use as start position their actual start
  // position but their already slightly moved position.
  //
  if (data.isMoving) {
    return true;
  }

  // Check whether the hovered object is movable, otherwise
  // there is nothing to move and we can fast-return.
  //
  const doc = documentManager.document();
  const selection = documentManager.selection();
  const hovered = selection.hovered();
  const hoveredMovable = getMovable(doc, hovered);
  if (!hovered || !hoveredMovable) {
    return false;
  }

  // Compute which objects should we move and their onMove() callback.
  //
  if (selection.isSelected(hovered)) {
    // If the hovered object is part of a larger selection, then we want to
    // move the whole selection
    data.movables = [];
    for (const s of selection.selected()) {
      const movable = getMovable(doc, s);
      if (movable) {
        data.movables.push(movable);
      }
    }
  } else {
    // If the hovered object is not currently selected, then we want to only
    // move the hovered object and set it as new selection.
    data.movables = [hoveredMovable];
    selection.setSelected([hovered]);
  }

  data.isMoving = true;
  return true;
}

function move(
  data: MoveData,
  documentManager: DocumentManager,
  delta: Vector2,
) {
  const selection = documentManager.selection();
  for (const movable of data.movables) {
    movable.onMove(delta, selection);
  }
  documentManager.stageChanges();
}

function end(data: MoveData, documentManager: DocumentManager) {
  data.isMoving = false;
  data.movables = [];
  documentManager.commitChanges();
}

export interface Mover {
  start: () => boolean; // Returns whether there is something to move
  move: (delta: Vector2) => void;
  end: () => void;
}

export function useMover(documentManager: DocumentManager): Mover {
  return useMemo(() => {
    const data = new MoveData();
    return {
      start: () => {
        return start(data, documentManager);
      },
      move: (delta: Vector2) => {
        move(data, documentManager, delta);
      },
      end: () => {
        end(data, documentManager);
      },
    };
  }, [documentManager]);
}
