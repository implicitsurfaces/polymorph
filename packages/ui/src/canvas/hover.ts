import { Vector2 } from "threejs-math";
import { Camera2 } from "./Camera2.ts";
import { Selectable } from "../Selection.ts";
import { DocumentManager } from "../DocumentManager.ts";
import { Document, isEdgeElement, Layer } from "../Document.ts";
import { getEdgeShapesAndControls } from "./drawEdges.ts";
import { controlPointRadius, pointRadius } from "./style.ts";

interface ClosestSelectable {
  selectable: Selectable | undefined;
  distance: number;
}

function findClosestSelectableInLayer(
  document: Document,
  camera: Camera2,
  layer: Layer,
  position: Vector2,
): ClosestSelectable {
  // Compute distance squared to closest point or control point center
  let closestDistanceSquared = Infinity;
  let closestSelectablePoint: Selectable | undefined = undefined;
  // For now, we only look for points
  for (const id of layer.elements) {
    const element = document.getElementFromId(id);
    if (element) {
      if (element.type === "Point") {
        const d = element.position.distanceToSquared(position);
        if (d < closestDistanceSquared) {
          closestDistanceSquared = d;
          closestSelectablePoint = { type: "Element", id: element.id };
        }
      } else if (isEdgeElement(element)) {
        // TODO: cache the controls from the draw call?
        const sc = getEdgeShapesAndControls(document, element);
        for (const cp of sc.controlPoints) {
          const d = cp.position.distanceToSquared(position);
          if (d < closestDistanceSquared) {
            closestDistanceSquared = d;
            closestSelectablePoint = {
              type: "SubElement",
              id: element.id,
              subName: cp.name,
            };
          }
        }
      }
    }
  }

  // Compute distance to closest point disk
  let closestDistance = Infinity;
  if (closestSelectablePoint) {
    let radius = pointRadius / camera.zoom;
    if (closestSelectablePoint.type === "SubElement") {
      radius = controlPointRadius;
    }
    if (closestDistanceSquared <= radius * radius) {
      // Position is inside the disk
      closestDistance = 0;
    } else {
      // Position is outside the disk
      closestDistance = Math.sqrt(closestDistanceSquared) - radius;
    }
  }
  return { selectable: closestSelectablePoint, distance: closestDistance };
}

function findClosestSelectableInDocument(
  document: Document,
  camera: Camera2,
  position: Vector2,
): ClosestSelectable {
  let closestDistance = Infinity;
  let selectable: Selectable | undefined = undefined;
  for (const id of document.layers) {
    const layer = document.getElementFromId<Layer>(id);
    if (layer) {
      const ce = findClosestSelectableInLayer(
        document,
        camera,
        layer,
        position,
      );
      if (ce.distance < closestDistance) {
        closestDistance = ce.distance;
        selectable = ce.selectable;
      }
    }
  }
  return { selectable: selectable, distance: closestDistance };
}

export function hover(
  documentManager: DocumentManager,
  camera: Camera2,
  mousePosition: Vector2,
  tolerance: number,
) {
  const doc = documentManager.document();
  const selection = documentManager.selection();
  const ce = findClosestSelectableInDocument(doc, camera, mousePosition);
  if (ce.selectable && ce.distance < tolerance) {
    selection.setHovered(ce.selectable);
  } else {
    selection.setHovered(undefined);
  }
}
