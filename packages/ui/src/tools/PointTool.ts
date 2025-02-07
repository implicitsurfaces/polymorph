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
    const layer = doc.getElement(selection.activeLayer(), Layer);
    if (!layer) {
      return;
    }
    const point = doc.createElementInLayer(Point, layer, {
      position: event.documentPosition,
    });
    selection.setHoveredElement(point.id);
    selection.setSelectedElements([point.id]);
    event.documentManager.commitChanges();
  }
}
