import { TriggerAction } from "./Action";
import { KeyboardShortcut } from "./KeyboardShortcut";
import icon from "../assets/tool-icons/point-to-point-distance.svg";

import { DocumentManager } from "../doc/DocumentManager";

import { Layer } from "../doc/Layer";
import { Point } from "../doc/Point";
import { PointToPointDistance } from "../doc/measures/PointToPointDistance";

export class AddDistanceAction extends TriggerAction {
  constructor() {
    super({
      name: "Add Distance Measure",
      icon: icon,
      shortcut: new KeyboardShortcut("D"),
    });
  }

  onTrigger(documentManager: DocumentManager) {
    // Check that the selection is composed of exactly two points
    const doc = documentManager.document();
    const selection = documentManager.selection();
    const layer = doc.getNode(selection.activeLayerId(), Layer);
    if (!layer) {
      return "No active layer.";
    }
    const selectedNodeIds = selection.selectedNodeIds();
    const points = doc.getNodes(selectedNodeIds, Point);
    if (selectedNodeIds.length != 2 || points.length !== 2) {
      return "Please select two points.";
    }
    const p1 = points[0];
    const p2 = points[1];

    // Check that a measure between these two points does not already exist
    for (const node of doc.nodes()) {
      if (node instanceof PointToPointDistance) {
        if (
          (node.startPoint === p1 && node.endPoint === p2) ||
          (node.startPoint === p2 && node.endPoint === p1)
        ) {
          return "A point-to-point distance measure between these two points already exists.";
        }
      }
    }

    // Create the measure and set it as selection
    const d = doc.createNode(PointToPointDistance, {
      layer: layer,
      startPoint: p1,
      endPoint: p2,
    });
    selection.setSelectedNodes([d]);
    documentManager.notifyChanges({ buildConstraints: true });
  }
}
