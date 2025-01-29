import { Vector2 } from "threejs-math";
import { CanvasPointerEvent } from "../canvas/events";

export interface Tool {
  readonly name: string;
  readonly icon: string;

  readonly onCanvasHover?: (event: CanvasPointerEvent) => void;
  readonly onCanvasClick?: (event: CanvasPointerEvent) => void;

  // The `onCanvasDragStart` callback must return a boolean to indicate
  // whether there is actually some drag action possible for the given
  // event.
  //
  // If there is something draggable, then the canvas will follow up by
  // calling `onCanvasDragMove` on mouse move, and `onCanvasDragEnd` on mouse
  // release.
  //
  // Otherwise, the canvas will follow up by calling `onCanvasClick` on mouse
  // release.
  //
  readonly onCanvasDragStart?: (event: CanvasPointerEvent) => boolean;
  readonly onCanvasDragMove?: (delta: Vector2) => void;
  readonly onCanvasDragEnd?: () => void;
}
