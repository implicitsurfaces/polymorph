import { Vector2 } from "threejs-math";
import { Camera2 } from "./Camera2.ts";
import { FillStyle, pointRadius, getNodeColor } from "./style.ts";

import { Selection } from "../Selection.ts";
import { Document, NodeId, Point } from "../Document.ts";

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
  const isHovered = selection.isHoveredNode(point.id);
  const isSelected = selection.isSelectedNode(point.id);
  const fillStyle = getNodeColor(isHovered, isSelected);
  drawDisk(ctx, point.getPosition(), pointRadius / camera.zoom, fillStyle);
}

export function drawPoints(
  ctx: CanvasRenderingContext2D,
  camera: Camera2,
  document: Document,
  nodes: Array<NodeId>,
  selection: Selection,
) {
  for (const id of nodes) {
    const point = document.getNode(id, Point);
    if (point) {
      drawPoint(ctx, camera, point, selection);
    }
  }
}
