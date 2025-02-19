import { Vector2 } from "threejs-math";
import { Camera2 } from "./Camera2";
import { FillStyle, pointRadius, getNodeColor } from "./style";

import { Selection } from "../Selection";
import { Point } from "../Document";

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
  const isHovered = selection.isHoveredNode(point);
  const isSelected = selection.isSelectedNode(point);
  const fillStyle = getNodeColor(isHovered, isSelected);
  drawDisk(ctx, point.position, pointRadius / camera.zoom, fillStyle);
}

export function drawPoints(
  ctx: CanvasRenderingContext2D,
  camera: Camera2,
  points: Point[],
  selection: Selection,
) {
  for (const point of points) {
    drawPoint(ctx, camera, point, selection);
  }
}
