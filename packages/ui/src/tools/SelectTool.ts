import { Vector2 } from "threejs-math";

import { Tool } from "./Tool";
import icon from "../assets/tool-icons/select.svg";

import { hover } from "../canvas/hover";
import { Mover } from "../canvas/move";
import { CanvasPointerEvent } from "../canvas/events";

export class SelectTool implements Tool {
  readonly name = "Select";
  readonly icon = icon;

  private mover: Mover | undefined = undefined;

  constructor() {}

  onCanvasHover(event: CanvasPointerEvent) {
    const toleranceInPx = 3;
    const toleranceInDocCoords = toleranceInPx / event.camera.zoom;
    hover(
      event.documentManager,
      event.camera,
      event.documentPosition,
      toleranceInDocCoords,
    );
  }

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
  }

  onCanvasDragStart(event: CanvasPointerEvent) {
    if (this.mover) {
      // This can happen in React strict mode, for which onCanvasDragStart may
      // be called twice before onCanvasDragEnd. In such scenario, we have to
      // return the same value as in the first call, that is, `true`.
      return true;
    }
    this.mover = new Mover(event.documentManager);
    const started = this.mover.start();
    if (!started) {
      this.mover = undefined;
    }
    return started;
  }

  onCanvasDragMove(delta: Vector2) {
    if (!this.mover) {
      return;
    }
    this.mover.move(delta);
  }

  onCanvasDragEnd() {
    if (!this.mover) {
      return;
    }
    this.mover.end();
    this.mover = undefined;
  }
}
