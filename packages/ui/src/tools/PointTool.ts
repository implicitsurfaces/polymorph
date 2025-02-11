import { Tool } from "./Tool";
import icon from "../assets/tool-icons/point.svg";

import { CanvasPointerEvent } from "../canvas/events";

import { Layer, Point } from "../Document";

export class PointTool implements Tool {
  readonly name = "Point";
  readonly icon = icon;

  constructor() {}

  onCanvasClick(event: CanvasPointerEvent) {
    const doc = event.documentManager.document();
    const selection = event.documentManager.selection();
    const layer = doc.getNode(selection.activeLayer(), Layer);
    if (!layer) {
      return;
    }
    const point = doc.createNodeInLayer(Point, layer, {
      position: event.documentPosition,
    });
    selection.setHoveredNode(point.id);
    selection.setSelectedNodes([point.id]);
    event.documentManager.commitChanges();
  }
}
