import { Vector2 } from "threejs-math";
import { Camera2 } from "./Camera2";
import { DocumentManager } from "../doc/DocumentManager";

export interface CanvasPointerEvent {
  readonly camera: Camera2;
  readonly viewPosition: Vector2;
  readonly documentPosition: Vector2;
  readonly documentManager: DocumentManager;
  readonly shiftKey: boolean;
}
