import { Vector2, Matrix3 } from "threejs-math";
import { Camera2 } from "./Camera2";
import { StrokeStyle } from "./style";

function moveTo(ctx: CanvasRenderingContext2D, p: Vector2) {
  ctx.moveTo(p.x, p.y);
}

function lineTo(ctx: CanvasRenderingContext2D, p: Vector2) {
  ctx.lineTo(p.x, p.y);
}

/**
 * If the line segment [p1, p2] is perfectly horizontal or perfectly vertical,
 * then snap it to the pixel grid for pixel-perfect rendering.
 */
function pixelSnap(p1: Vector2, p2: Vector2) {
  if (Math.abs(p1.x - p2.x) < 0.01) {
    p1.round();
    p2.round();
    p1.x -= 0.5;
    p2.x -= 0.5;
  } else if (Math.abs(p1.y - p2.y) < 0.01) {
    p1.round();
    p2.round();
    p1.y -= 0.5;
    p2.y -= 0.5;
  }
}

// Transform p1 and p2 to view coords, pixel snap them, then moveTo(p1) and lineTo(p2)
function drawGridLine(
  ctx: CanvasRenderingContext2D,
  documentToView: Matrix3,
  p1: Vector2,
  p2: Vector2,
) {
  p1.applyMatrix3(documentToView);
  p2.applyMatrix3(documentToView);
  pixelSnap(p1, p2);
  moveTo(ctx, p1);
  lineTo(ctx, p2);
}

function drawGridCells(
  ctx: CanvasRenderingContext2D,
  documentToView: Matrix3,
  xMin: number,
  xMax: number,
  yMin: number,
  yMax: number,
  size: number,
  strokeStyle: StrokeStyle,
) {
  // Get document coords of the first visible horizontal and vertical grid line,
  // as well as how many of them are visible.
  const xStart = Math.floor(xMin / size) * size;
  const xNum = Math.floor((xMax - xMin) / size) + 2;
  const yStart = Math.floor(yMin / size) * size;
  const yNum = Math.floor((yMax - yMin) / size) + 2;

  ctx.beginPath();

  // Vertical lines
  for (let i = 0; i < xNum; ++i) {
    const x = xStart + i * size;
    drawGridLine(
      ctx,
      documentToView,
      new Vector2(x, yMin),
      new Vector2(x, yMax),
    );
  }

  // Horizontal lines
  for (let i = 0; i < yNum; ++i) {
    const y = yStart + i * size;
    drawGridLine(
      ctx,
      documentToView,
      new Vector2(xMin, y),
      new Vector2(xMax, y),
    );
  }

  ctx.lineWidth = 1;
  ctx.strokeStyle = strokeStyle;
  ctx.stroke();
}

function drawGridAxes(
  ctx: CanvasRenderingContext2D,
  documentToView: Matrix3,
  xMin: number,
  xMax: number,
  yMin: number,
  yMax: number,
  strokeStyle: StrokeStyle,
) {
  ctx.beginPath();
  drawGridLine(ctx, documentToView, new Vector2(xMin, 0), new Vector2(xMax, 0));
  drawGridLine(ctx, documentToView, new Vector2(0, yMin), new Vector2(0, yMax));
  ctx.lineWidth = 1;
  ctx.strokeStyle = strokeStyle;
  ctx.stroke();
}

export function drawGrid(ctx: CanvasRenderingContext2D, camera: Camera2) {
  const documentToView = camera.viewMatrix();
  const viewToDocument = documentToView.clone().invert();
  const w = camera.canvasSize.x;
  const h = camera.canvasSize.y;
  const axesColor = "#808080";
  const majorColor = "#b2b2b2";
  const minorColor = "#cccccc";

  // Size of minor and major grid cells in document coords
  const minorSize = Math.pow(10, Math.ceil(0.6 - Math.log10(camera.zoom)));
  const majorSize = minorSize * 10;

  // Express view rectangle in document coordinates
  const p1 = new Vector2(0, 0).applyMatrix3(viewToDocument);
  const p2 = new Vector2(w, 0).applyMatrix3(viewToDocument);
  const p3 = new Vector2(w, h).applyMatrix3(viewToDocument);
  const p4 = new Vector2(0, h).applyMatrix3(viewToDocument);

  // Get min/max in document coordinates
  const xMin = Math.min(p1.x, p2.x, p3.x, p4.x);
  const xMax = Math.max(p1.x, p2.x, p3.x, p4.x);
  const yMin = Math.min(p1.y, p2.y, p3.y, p4.y);
  const yMax = Math.max(p1.y, p2.y, p3.y, p4.y);

  drawGridCells(
    ctx,
    documentToView,
    xMin,
    xMax,
    yMin,
    yMax,
    minorSize,
    minorColor,
  );
  drawGridCells(
    ctx,
    documentToView,
    xMin,
    xMax,
    yMin,
    yMax,
    majorSize,
    majorColor,
  );
  drawGridAxes(ctx, documentToView, xMin, xMax, yMin, yMax, axesColor);
}
