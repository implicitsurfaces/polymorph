import { Tool } from "./Tool";
import icon from "../assets/tool-icons/select.svg";

import { hover } from "../canvas/hover";
import { getMover } from "../canvas/move";
import { CanvasPointerEvent } from "../canvas/events";

export const SelectTool: Tool = {
  name: "Select",
  icon: icon,
  onCanvasHover(event: CanvasPointerEvent) {
    const toleranceInPx = 3;
    const toleranceInDocCoords = toleranceInPx / event.camera.zoom;
    hover(
      event.documentManager,
      event.camera,
      event.documentPosition,
      toleranceInDocCoords,
    );
  },
  onCanvasClick(event: CanvasPointerEvent) {
    const selection = event.documentManager.selection();
    const hovered = selection.hovered();
    if (hovered) {
      if (event.shiftKey) {
        selection.toggleSelected(hovered);
      } else {
        selection.setSelected([hovered]);
      }
    }
  },
  onCanvasDrag(event: CanvasPointerEvent) {
    return getMover(event.documentManager);
  },
};
