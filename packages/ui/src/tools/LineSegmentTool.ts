import { Vector2 } from "threejs-math";

import { Tool } from "./Tool";
import { KeyboardShortcut } from "../actions/KeyboardShortcut";
import icon from "../assets/tool-icons/line-segment.svg";

import { Camera2 } from "../canvas/Camera2";
import { CanvasPointerEvent } from "../canvas/events";
import { hover, hoverFromCanvas } from "../canvas/hover";

import { DocumentManager } from "../doc/DocumentManager";
import { Node, NodeId } from "../doc/Node";
import { Layer } from "../doc/Layer";
import { Point } from "../doc/Point";
import { LineSegment } from "../doc/edges/LineSegment";

// The tool works equivalently either via two separate clicks or one drag:
//
// 1. On the first step (click or drag start), we determine the start point of
// the line segment, and create a temporary line segment that for now starts
// and ends at the same point.
//
// 2. On mouse hover between the steps (or drag move), we update the end point
//    of the temporary line segment. It could either be an existing point,
//    or a new temporary end point of which we update the position.
//
// 3. On the second step (click or drag end), we commit as final the above
//    temporary changes.

interface FirstStepData {
  readonly documentManager: DocumentManager;
  readonly camera: Camera2;
  readonly startPosition: Vector2;
  readonly lineSegmentId: NodeId;
  isEndPointPreexisting: boolean;
}

// Returns the ID of the currently hovered point if any.
//
function getHoveredPoint(documentManager: DocumentManager): Point | undefined {
  const doc = documentManager.document();
  const selection = documentManager.selection();
  const hoveredNodeId = selection.hoveredNodeId();
  return doc.getNode(hoveredNodeId, Point);
}

// Returns the ID of the currently hovered point if any, otherwise
// creates a new point and returns its ID.
//
function getOrCreatePoint(
  documentManager: DocumentManager,
  position: Vector2,
  layer: Layer,
): Point {
  const hoveredPoint = getHoveredPoint(documentManager);
  if (hoveredPoint !== undefined) {
    return hoveredPoint;
  } else {
    // Create new point, set it as hovered, and return it
    const doc = documentManager.document();
    const selection = documentManager.selection();
    const point = doc.createNode(Point, {
      layer: layer,
      position: position,
    });
    selection.setHoveredNode(point);
    return point;
  }
}

export class LineSegmentTool extends Tool {
  constructor() {
    super({
      name: "Line Segment",
      icon: icon,
      shortcut: new KeyboardShortcut("L"),
    });
  }

  private firstStepData: FirstStepData | undefined;

  private reset() {
    this.firstStepData = undefined;
  }

  private doFirstStep(event: CanvasPointerEvent) {
    if (this.firstStepData) {
      return true;
    }

    // Get active layer
    const documentManager = event.documentManager;
    const doc = documentManager.document();
    const selection = documentManager.selection();
    const layer = doc.getNode(selection.activeLayerId(), Layer);
    if (!layer) {
      return false;
    }

    // Get or create start point
    const position = event.documentPosition;
    const startPoint = getOrCreatePoint(documentManager, position, layer);

    // Create temporary line segment.
    // For now its start point and end point are the same.
    // We set it as selected so that its properties already appear in
    // the Properties panel, giving feedback to the user that the first step
    // happens.
    const lineSegment = doc.createNode(LineSegment, {
      layer: layer,
      startPoint: startPoint,
      endPoint: startPoint,
    });
    selection.setSelectedNodes([lineSegment]);

    // Store data needed for subsequent steps and stage changes.
    this.firstStepData = {
      documentManager: documentManager,
      camera: event.camera,
      startPosition: position,
      lineSegmentId: lineSegment.id,
      isEndPointPreexisting: true, // [1]
    };
    // [1] Initially, the start point and the end point are the same,
    // and we always consider the start point to be preexisting, even
    // if it was created during the first step. Indeed, the start point
    // must never be deleted in the HoverBetweenSteps.
    documentManager.notifyChanges({ commit: false });
    return true;
  }

  private doHoverBetweenSteps(position: Vector2) {
    if (!this.firstStepData) {
      return;
    }

    // Get active layer
    const documentManager = this.firstStepData.documentManager;
    const doc = documentManager.document();
    const selection = documentManager.selection();
    const layer = doc.getNode(selection.activeLayerId(), Layer);
    if (!layer) {
      this.reset();
      return;
    }

    // Get line segment
    const lineSegment = doc.getNode(
      this.firstStepData.lineSegmentId,
      LineSegment,
    );
    if (!lineSegment) {
      this.reset();
      return;
    }

    // Update hovered object. We use a filter to prevent hovering a
    // temporary end point, if any.
    const tolerance = undefined; // Use default tolerance
    const filter = this.firstStepData.isEndPointPreexisting
      ? undefined
      : (node: Node) => {
          return node !== lineSegment.endPoint;
        };
    hover(
      this.firstStepData.documentManager,
      this.firstStepData.camera,
      position,
      tolerance,
      filter,
    );

    // Update end point.
    const hoveredPoint = getHoveredPoint(documentManager);
    if (hoveredPoint === undefined) {
      if (this.firstStepData.isEndPointPreexisting) {
        // Change from pre-existing endpoint to temporary new endpoint
        const point = doc.createNode(Point, {
          layer: layer,
          position: position,
        });
        lineSegment.endPoint = point;
        this.firstStepData.isEndPointPreexisting = false;
      } else {
        // Update the position of the temporary new endpoint
        lineSegment.endPoint.position = position;
      }
    } else {
      if (this.firstStepData.isEndPointPreexisting) {
        // Update which pre-existing endpoint the line segment refers to
        lineSegment.endPoint = hoveredPoint;
      } else {
        // Update from temporary new endpoint to pre-existing endpoint
        const tmpPoint = lineSegment.endPoint;
        lineSegment.endPoint = hoveredPoint;
        doc.removeNode(tmpPoint);
        this.firstStepData.isEndPointPreexisting = true;
      }
    }

    documentManager.notifyChanges({ commit: false });
  }

  // Get or create the end point, create the line segment, and set it as
  // selection.
  private doSecondStep() {
    if (!this.firstStepData) {
      return;
    }
    this.firstStepData.documentManager.notifyChanges({ commit: true });
    this.reset();
  }

  onCanvasHover(event: CanvasPointerEvent) {
    if (!this.firstStepData) {
      hoverFromCanvas(event);
    } else {
      this.doHoverBetweenSteps(event.documentPosition);
    }
  }

  onCanvasClick(event: CanvasPointerEvent) {
    if (!this.firstStepData) {
      this.doFirstStep(event);
    } else {
      this.doSecondStep(/*event.documentPosition*/);
    }
  }

  onCanvasDragStart(event: CanvasPointerEvent): boolean {
    return this.doFirstStep(event);
  }

  onCanvasDragMove(delta: Vector2) {
    if (!this.firstStepData) {
      return;
    }
    const position = this.firstStepData.startPosition.clone().add(delta);
    this.doHoverBetweenSteps(position);
  }

  onCanvasDragEnd() {
    if (!this.firstStepData) {
      return;
    }
    this.doSecondStep();
  }
}
