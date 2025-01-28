import { CanvasPointerEvent } from "../canvas/events";

export interface Tool {
  readonly name: string;
  readonly icon: string;
  readonly onCanvasHover?: (event: CanvasPointerEvent) => void;
  readonly onCanvasClick?: (event: CanvasPointerEvent) => void;
}
