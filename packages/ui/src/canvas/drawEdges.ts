import { Vector2 } from "threejs-math";
import { Camera2 } from "./Camera2.ts";
import {
  StrokeStyle,
  controlPointRadius,
  edgeWidth,
  getElementColor,
  getControlColor,
} from "./style.ts";

import { drawDisk } from "./drawPoints.ts";
import { Selection, Selectable } from "../Selection.ts";
import {
  Point,
  Document,
  ElementId,
  EdgeElement,
  isEdgeElement,
} from "../Document.ts";

export interface CanvasShapeBase {
  type: string;
}

export interface CanvasArc extends CanvasShapeBase {
  type: "Arc";
  center: Vector2;
  radius: number;
  startAngle: number;
  endAngle: number;
  isCounterClockwise: boolean;
}

export interface CanvasLineSegment extends CanvasShapeBase {
  type: "LineSegment";
  startPoint: Vector2;
  endPoint: Vector2;
}

export type CanvasGeneralizedArc = CanvasArc | CanvasLineSegment;
export type CanvasShape = CanvasGeneralizedArc;

interface EdgeStyle {
  lineWidth: number;
  strokeStyle: StrokeStyle;
}

function drawEdgeBegin(ctx: CanvasRenderingContext2D) {
  ctx.beginPath();
}

function drawEdgeEnd(ctx: CanvasRenderingContext2D, edgeStyle: EdgeStyle) {
  ctx.lineWidth = edgeStyle.lineWidth;
  ctx.strokeStyle = edgeStyle.strokeStyle;
  ctx.stroke();
}

function drawLineSegment(
  ctx: CanvasRenderingContext2D,
  lineSegment: CanvasLineSegment,
  edgeStyle: EdgeStyle,
) {
  drawEdgeBegin(ctx);
  ctx.moveTo(lineSegment.startPoint.x, lineSegment.startPoint.y);
  ctx.lineTo(lineSegment.endPoint.x, lineSegment.endPoint.y);
  drawEdgeEnd(ctx, edgeStyle);
}

function drawArc(
  ctx: CanvasRenderingContext2D,
  arc: CanvasArc,
  edgeStyle: EdgeStyle,
) {
  drawEdgeBegin(ctx);
  ctx.arc(
    arc.center.x,
    arc.center.y,
    arc.radius,
    arc.startAngle,
    arc.endAngle,
    arc.isCounterClockwise,
  );
  drawEdgeEnd(ctx, edgeStyle);
}

function drawShape(
  ctx: CanvasRenderingContext2D,
  shape: CanvasShape,
  edgeStyle: EdgeStyle,
) {
  switch (shape.type) {
    case "Arc":
      drawArc(ctx, shape, edgeStyle);
      break;
    case "LineSegment":
      drawLineSegment(ctx, shape, edgeStyle);
      break;
  }
}

function getLineSegment(
  startPoint: Vector2,
  endPoint: Vector2,
): CanvasLineSegment {
  return { type: "LineSegment", startPoint: startPoint, endPoint: endPoint };
}

function getGeneralizedArcFromStartTangent(
  startPoint: Vector2,
  endPoint: Vector2,
  tangent: Vector2,
): CanvasGeneralizedArc {
  // Fallback to line segment in degenerate cases (collinear)
  const chord = endPoint.clone().sub(startPoint);
  const s_ = chord.cross(tangent);
  if (s_ === 0) {
    return { type: "LineSegment", startPoint: startPoint, endPoint: endPoint };
  }

  // Compute center
  const tangentLength = tangent.length(); // guaranteed > 0 otherwise s_ == 0
  const s = s_ / tangentLength;
  const c = chord.dot(tangent) / tangentLength;
  const bulge = -s / (c + Math.sqrt(c * c + s * s));
  const chordPerp = new Vector2(-chord.y, chord.x);
  const bb = (bulge - 1 / bulge) / 4;
  const midPoint = startPoint.clone().add(endPoint).multiplyScalar(0.5);
  const center = midPoint.clone().sub(chordPerp.multiplyScalar(bb));

  return {
    type: "Arc",
    center: center,
    radius: center.distanceTo(startPoint),
    startAngle: startPoint.clone().sub(center).angle(),
    endAngle: endPoint.clone().sub(center).angle(),
    isCounterClockwise: s_ > 0,
  };
}

function getCCurveShapes(
  startPoint: Vector2,
  endPoint: Vector2,
  controlPoint: Vector2,
): Array<CanvasGeneralizedArc> {
  const w0 = controlPoint.distanceTo(endPoint);
  const w1 = controlPoint.distanceTo(startPoint);
  const w2 = startPoint.distanceTo(endPoint);

  const junction = startPoint
    .clone()
    .multiplyScalar(w0)
    .add(endPoint.clone().multiplyScalar(w1))
    .add(controlPoint.clone().multiplyScalar(w2))
    .divideScalar(w0 + w1 + w2);

  const startTangent = controlPoint.clone().sub(startPoint);
  const endTangent = controlPoint.clone().sub(endPoint);

  return [
    getGeneralizedArcFromStartTangent(startPoint, junction, startTangent),
    getGeneralizedArcFromStartTangent(endPoint, junction, endTangent),
  ];
}

function lineLineIntersection(
  p0: Vector2,
  v0: Vector2,
  p1: Vector2,
  v1: Vector2,
) {
  const crossDir = v0.cross(v1);
  const diffPoint = p1.clone().sub(p0);
  const param = diffPoint.cross(v1) / crossDir;
  return p0.clone().add(v0.clone().multiplyScalar(param));
}

function getMidpoint(a: Vector2, b: Vector2) {
  return a.clone().add(b).multiplyScalar(0.5);
}

function biarcLocusCenter(
  p1: Vector2,
  tangent1: Vector2,
  p2: Vector2,
  tangent2: Vector2,
) {
  const u1 = tangent1.clone().normalize();
  const u2 = tangent2.clone().normalize();

  const p3 = p1.clone().add(u1);
  const p4 = p2.clone().add(u2);

  const m1 = getMidpoint(p1, p2);
  const m2 = getMidpoint(p3, p4);

  const p1p2 = p2.clone().sub(p1);
  const p3p4 = p4.clone().sub(p3);

  const v1 = new Vector2(-p1p2.y, p1p2.x);
  const v2 = new Vector2(-p3p4.y, p3p4.x);

  return lineLineIntersection(m1, v1, m2, v2);
}

function getSCurveShapes(
  startPoint: Vector2,
  endPoint: Vector2,
  startControlPoint: Vector2,
  endControlPoint: Vector2,
): Array<CanvasGeneralizedArc> {
  const tangent1 = startControlPoint.clone().sub(startPoint);
  const tangent2 = endPoint.clone().sub(endControlPoint);

  const locusCenter = biarcLocusCenter(
    startPoint,
    tangent1,
    endPoint,
    tangent2,
  );

  // XXX: is it normal that positionRatio is always equal to 0.5
  // if the tangents are normalized?
  const positionRatio =
    tangent1.length() / (tangent1.length() + tangent2.length());
  const chord = endPoint.clone().sub(startPoint);

  const positionOnChord = startPoint
    .clone()
    .add(chord.clone().multiplyScalar(positionRatio));

  const junctionDir = positionOnChord.clone().sub(locusCenter).normalize();
  const junction = locusCenter
    .clone()
    .add(junctionDir.clone().multiplyScalar(locusCenter.distanceTo(endPoint)));

  return [
    getGeneralizedArcFromStartTangent(startPoint, junction, tangent1),
    getGeneralizedArcFromStartTangent(endPoint, junction, tangent2.negate()),
  ];
}

export interface ControlPoint {
  readonly name: string;
  readonly position: Vector2;
  readonly onMove: (delta: Vector2, selection: Selection) => void;
}

export interface EdgeShapesAndControls {
  shapes: Array<CanvasShape>;
  controlPoints: Array<ControlPoint>;
  tangents: Array<CanvasLineSegment>;
}

export function getEdgeShapesAndControls(
  doc: Document,
  element: EdgeElement,
): EdgeShapesAndControls {
  const res: EdgeShapesAndControls = {
    shapes: [],
    controlPoints: [],
    tangents: [],
  };

  const startPoint = doc.getElementFromId<Point>(element.startPoint);
  const endPoint = doc.getElementFromId<Point>(element.endPoint);
  if (!startPoint || !endPoint) {
    return res;
  }
  const startPos = startPoint.position;
  const endPos = endPoint.position;

  switch (element.type) {
    case "LineSegment": {
      res.shapes.push(getLineSegment(startPos, endPos));
      break;
    }
    case "ArcFromStartTangent": {
      const tangent = element.tangent;
      const cpAbsPos = tangent.clone().add(startPos);
      res.shapes.push(
        getGeneralizedArcFromStartTangent(startPos, endPos, tangent),
      );
      res.controlPoints.push({
        name: "tangent",
        position: cpAbsPos,
        onMove: (delta: Vector2, selection: Selection) => {
          if (!selection.isSelectedElement(startPoint.id)) {
            element.tangent = tangent.clone().add(delta);
          }
        },
      });
      res.tangents.push(getLineSegment(startPos, cpAbsPos));
      break;
    }
    case "CCurve": {
      const cpRelPoint = (function () {
        switch (element.mode) {
          case "startTangent":
            return startPoint;
          case "endTangent":
            return endPoint;
        }
        return undefined;
      })();
      const cpRelPos = element.controlPoint;
      const cpAbsPos = cpRelPoint
        ? cpRelPos.clone().add(cpRelPoint.position)
        : cpRelPos;

      const shapes_ = getCCurveShapes(startPos, endPos, cpAbsPos);
      for (const shape of shapes_) {
        res.shapes.push(shape);
      }
      res.controlPoints.push({
        name: "controlPoint",
        position: cpAbsPos,
        onMove: (delta: Vector2, selection: Selection) => {
          const relId = cpRelPoint?.id;
          if (!relId || !selection.isSelectedElement(relId)) {
            element.controlPoint = cpRelPos.clone().add(delta);
          }
        },
      });
      res.tangents.push(getLineSegment(startPos, cpAbsPos));
      res.tangents.push(getLineSegment(endPos, cpAbsPos));
      break;
    }
    case "SCurve": {
      const startCpRelPos = element.startControlPoint;
      const endCpRelPos = element.endControlPoint;
      let startCpRelPoint = undefined;
      let endCpRelPoint = undefined;
      let startCpAbsPos = startCpRelPos;
      let endCpAbsPos = endCpRelPos;
      if (element.mode === "tangent") {
        startCpRelPoint = startPoint;
        endCpRelPoint = endPoint;
        startCpAbsPos = startCpRelPos.clone().add(startPos);
        endCpAbsPos = endCpRelPos.clone().add(endPos);
      }
      const shapes_ = getSCurveShapes(
        startPos,
        endPos,
        startCpAbsPos,
        endCpAbsPos,
      );
      for (const shape of shapes_) {
        res.shapes.push(shape);
      }
      res.controlPoints.push({
        name: "startControlPoint",
        position: startCpAbsPos,
        onMove: (delta: Vector2, selection: Selection) => {
          const relId = startCpRelPoint?.id;
          if (!relId || !selection.isSelectedElement(relId)) {
            element.startControlPoint = startCpRelPos.clone().add(delta);
          }
        },
      });
      res.controlPoints.push({
        name: "endControlPoint",
        position: endCpAbsPos,
        onMove: (delta: Vector2, selection: Selection) => {
          const relId = endCpRelPoint?.id;
          if (!relId || !selection.isSelectedElement(relId)) {
            element.endControlPoint = endCpRelPos.clone().add(delta);
          }
        },
      });
      res.tangents.push(getLineSegment(startPos, startCpAbsPos));
      res.tangents.push(getLineSegment(endPos, endCpAbsPos));
      break;
    }
  }
  return res;
}

export function drawEdges(
  ctx: CanvasRenderingContext2D,
  camera: Camera2,
  document: Document,
  elements: Array<ElementId>,
  selection: Selection,
) {
  const cpRadius = controlPointRadius / camera.zoom;
  const edgeWidth_ = edgeWidth / camera.zoom;
  const edgeStyle = { lineWidth: edgeWidth_, strokeStyle: getElementColor() };
  const tangentStyle = {
    lineWidth: edgeWidth_,
    strokeStyle: getControlColor(),
  };
  for (const id of elements) {
    const element = document.getElementFromId(id);
    if (element && isEdgeElement(element)) {
      const isEdgeHovered = selection.isHoveredElement(id);
      const isEdgeSelected = selection.isSelectedElement(id);
      edgeStyle.strokeStyle = getElementColor(isEdgeHovered, isEdgeSelected);
      const sc = getEdgeShapesAndControls(document, element);
      for (const shape of sc.shapes) {
        drawShape(ctx, shape, edgeStyle);
      }
      for (const lineSegment of sc.tangents) {
        drawLineSegment(ctx, lineSegment, tangentStyle);
      }
      for (const cp of sc.controlPoints) {
        const selectable: Selectable = {
          type: "SubElement",
          id: id,
          subName: cp.name,
        };
        const isCpHovered = selection.isHovered(selectable);
        const isCpSelected = selection.isSelected(selectable);
        const cpColor = getControlColor(isCpHovered, isCpSelected);
        drawDisk(ctx, cp.position, cpRadius, cpColor);
      }
    }
  }
}
