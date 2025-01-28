import { Vector2 } from "threejs-math";
import { CanvasPointerEvent } from "../canvas/events";

export interface CanvasDragAction {
  readonly start: () => boolean; // return whether the drag action was actually started
  readonly move: (delta: Vector2) => void;
  readonly end: () => void;
}

export interface Tool {
  readonly name: string;
  readonly icon: string;
  readonly onCanvasHover?: (event: CanvasPointerEvent) => void;
  readonly onCanvasClick?: (event: CanvasPointerEvent) => void;
  readonly onCanvasDrag?: (
    event: CanvasPointerEvent,
  ) => CanvasDragAction | undefined;
}
