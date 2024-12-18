import { Selection } from "../Selection.ts";
import { Document, ElementId, Layer } from "../Document.ts";
import { DocumentManager } from "../DocumentManager.ts";
import { Camera2 } from "./Camera2.ts";
import { FillStyle } from "./style.ts";
import { drawEdges } from "./drawEdges.ts";
import { drawPoints } from "./drawPoints.ts";
import { drawGrid } from "./drawGrid.ts";

function drawBackground(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  fillStyle: FillStyle,
) {
  ctx.beginPath();
  ctx.fillStyle = fillStyle;
  ctx.fillRect(0, 0, width, height);
}

function initializeViewTransform(
  ctx: CanvasRenderingContext2D,
  camera: Camera2,
) {
  const e = camera.viewMatrix().elements;
  ctx.setTransform(e[0], e[1], e[3], e[4], e[6], e[7]);
}

function drawDocument(
  ctx: CanvasRenderingContext2D,
  camera: Camera2,
  document: Document,
  selection: Selection,
) {
  document.layers.forEach((id: ElementId) => {
    const layer = document.getElementFromId<Layer>(id);
    if (layer) {
      // Note: we use two passes since we want to draw all points on top of
      // edges, regardless of layer order.
      drawEdges(ctx, camera, document, layer.elements, selection);
      drawPoints(ctx, camera, document, layer.elements, selection);
    }
  });
}

export function draw(
  canvas: HTMLCanvasElement,
  camera: Camera2,
  documentManager: DocumentManager,
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }
  ctx.resetTransform();
  drawBackground(ctx, canvas.width, canvas.height, "#e0e0e0");
  drawGrid(ctx, camera);
  initializeViewTransform(ctx, camera);
  const document = documentManager.document();
  const selection = documentManager.selection();
  drawDocument(ctx, camera, document, selection);
}
