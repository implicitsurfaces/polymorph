import { Vector2 } from "threejs-math";

import { Tool } from "./Tool";
import icon from "../assets/tool-icons/line-segment.svg";

import { Camera2 } from "../canvas/Camera2";
import { CanvasPointerEvent } from "../canvas/events";
import { hover, hoverFromCanvas } from "../canvas/hover";

import { DocumentManager } from "../DocumentManager";
import { ElementId, Layer, Point, LineSegment } from "../Document";

interface FirstStepData {
  readonly documentManager: DocumentManager;
  readonly pointId: ElementId;
  readonly position: Vector2;
  readonly camera: Camera2;
}

export class LineSegmentTool implements Tool {
  readonly name = "Line Segment";
  readonly icon = icon;

  private firstStepData: FirstStepData | undefined;
  private dragMovePosition: Vector2 = new Vector2(0, 0);

  private reset() {
    this.firstStepData = undefined;
  }

  constructor() {}

  onCanvasHover(event: CanvasPointerEvent) {
    hoverFromCanvas(event);
  }

  private getOrCreatePoint(
    documentManager: DocumentManager,
    position: Vector2,
    layer: Layer,
  ): [ElementId, boolean] {
    const doc = documentManager.document();
    const selection = documentManager.selection();
    const hoveredElementId = selection.hoveredElement();
    const hoveredElement = doc.getElementFromId(hoveredElementId);
    if (hoveredElement?.type === "Point") {
      // Return hovered point
      return [hoveredElement.id, false];
    } else {
      // Create new point, set it as hovered, and return it
      const point = doc.createElementInLayer(Point, layer, {
        position: position,
      });
      selection.setHoveredElement(point.id);
      return [point.id, true];
    }
  }

  // Get or create the start point and set it as selection
  private onFirstStep(event: CanvasPointerEvent) {
    if (this.firstStepData) {
      return true;
    }

    const doc = event.documentManager.document();
    const selection = event.documentManager.selection();
    const layer = doc.getElementFromId<Layer>(selection.activeLayer());
    if (!layer) {
      return false;
    }

    const [pointId, isCreated] = this.getOrCreatePoint(
      event.documentManager,
      event.documentPosition,
      layer,
    );
    selection.setSelectedElements([pointId]);
    if (isCreated) {
      event.documentManager.commitChanges();
    }
    this.firstStepData = {
      documentManager: event.documentManager,
      pointId: pointId,
      position: event.documentPosition,
      camera: event.camera,
    };
    return true;
  }

  // Get or create the end point, create the line segment, and set it as
  // selection.
  private onSecondStep(position: Vector2) {
    if (!this.firstStepData) {
      return;
    }

    const documentManager = this.firstStepData.documentManager;
    const doc = documentManager.document();
    const selection = documentManager.selection();
    const layer = doc.getElementFromId<Layer>(selection.activeLayer());
    if (!layer) {
      this.reset();
      return;
    }

    // Get or create the end point
    const [endPointId] = this.getOrCreatePoint(
      documentManager,
      position,
      layer,
    );
    if (!endPointId) {
      this.reset();
      return;
    }
    // Create the line segment
    const lineSegment = doc.createElementInLayer(LineSegment, layer, {
      startPoint: this.firstStepData.pointId,
      endPoint: endPointId,
    });
    documentManager.commitChanges();
    selection.setHoveredElement(endPointId);
    selection.setSelectedElements([lineSegment.id]);
    this.reset();
  }

  onCanvasClick(event: CanvasPointerEvent) {
    if (!this.firstStepData) {
      this.onFirstStep(event);
    } else {
      this.onSecondStep(event.documentPosition);
    }
  }

  onCanvasDragStart(event: CanvasPointerEvent) {
    const started = this.onFirstStep(event);
    if (this.firstStepData) {
      this.dragMovePosition = this.firstStepData.position;
    }
    return started;
  }

  onCanvasDragMove(delta: Vector2) {
    if (!this.firstStepData) {
      return;
    }
    this.dragMovePosition = this.firstStepData.position.clone().add(delta);
    hover(
      this.firstStepData.documentManager,
      this.firstStepData.camera,
      this.dragMovePosition,
    );
  }

  onCanvasDragEnd() {
    if (!this.firstStepData) {
      return;
    }
    this.onSecondStep(this.dragMovePosition);
  }
}
