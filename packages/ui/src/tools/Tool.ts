import { Vector2 } from "threejs-math";
import { CanvasPointerEvent } from "../canvas/events";
import { Action, ActionProps } from "../actions/Action";

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface ToolProps extends ActionProps {}

export class Tool extends Action {
  constructor(props: ToolProps) {
    super(props);
  }

  onCanvasHover?(event: CanvasPointerEvent): void;
  onCanvasClick?(event: CanvasPointerEvent): void;

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
  onCanvasDragStart?(event: CanvasPointerEvent): boolean;
  onCanvasDragMove?(delta: Vector2): void;
  onCanvasDragEnd?(): void;
}
