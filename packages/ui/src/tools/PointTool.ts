import { Tool } from "./Tool";
import { KeyboardShortcut } from "../actions/KeyboardShortcut";
import icon from "../assets/tool-icons/point.svg";

import { CanvasPointerEvent } from "../canvas/events";

import { Layer, Point } from "../doc/Document";

export class PointTool extends Tool {
  constructor() {
    super({
      name: "Point",
      icon: icon,
      shortcut: new KeyboardShortcut("P"),
    });
  }

  onCanvasClick(event: CanvasPointerEvent) {
    const doc = event.documentManager.document();
    const selection = event.documentManager.selection();
    const layer = doc.getNode(selection.activeLayerId(), Layer);
    if (!layer) {
      return;
    }
    const point = doc.createNode(Point, {
      layer: layer,
      position: event.documentPosition,
    });
    selection.setHoveredNode(point);
    selection.setSelectedNodes([point]);
    event.documentManager.notifyChanges();
  }
}
