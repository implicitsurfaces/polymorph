import { Vector2 } from "threejs-math";
import { Camera2 } from "./Camera2.ts";
import { FillStyle, pointRadius, getElementColor } from "./style.ts";

import { Selection } from "../Selection.ts";
import { Document, ElementId, Point } from "../Document.ts";

export function drawDisk(
  ctx: CanvasRenderingContext2D,
  position: Vector2,
  radius: number,
  fillStyle: FillStyle,
) {
  ctx.beginPath();
  ctx.arc(position.x, position.y, radius, 0, 2 * Math.PI);
  ctx.fillStyle = fillStyle;
  ctx.fill();
}

export function drawPoint(
  ctx: CanvasRenderingContext2D,
  camera: Camera2,
  point: Point,
  selection: Selection,
) {
  const isHovered = selection.isHoveredElement(point.id);
  const isSelected = selection.isSelectedElement(point.id);
  const fillStyle = getElementColor(isHovered, isSelected);
  drawDisk(ctx, point.position, pointRadius / camera.zoom, fillStyle);
}

export function drawPoints(
  ctx: CanvasRenderingContext2D,
  camera: Camera2,
  document: Document,
  elements: Array<ElementId>,
  selection: Selection,
) {
  for (const id of elements) {
    const element = document.getElementFromId(id);
    if (element && element.type === "Point") {
      drawPoint(ctx, camera, element, selection);
    }
  }
}
