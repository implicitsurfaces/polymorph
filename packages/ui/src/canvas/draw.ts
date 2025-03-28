import { Selection } from "../Selection";
import { Document, Layer, EdgeNode, Point, MeasureNode } from "../Document";
import { DocumentManager } from "../DocumentManager";
import { Camera2 } from "./Camera2";
import { FillStyle, backgroundColor } from "./style";
import { drawEdges } from "./drawEdges";
import { drawPoints } from "./drawPoints";
import { drawGrid } from "./drawGrid";
import { drawMeasures } from "./drawMeasures";

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
  doc: Document,
  selection: Selection,
) {
  for (const id of doc.layers) {
    const layer = doc.getNode(id, Layer);
    if (layer) {
      // Note: we use two passes since we want to draw all points on top of
      // edges, regardless of layer order.
      drawEdges(
        ctx,
        camera,
        doc.getNodes(layer.data.nodeIds, EdgeNode),
        selection,
      );
      drawPoints(
        ctx,
        camera,
        doc.getNodes(layer.data.nodeIds, Point),
        selection,
      );
      drawMeasures(
        ctx,
        camera,
        doc.getNodes(layer.data.nodeIds, MeasureNode),
        selection,
      );
    }
  }
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
  drawBackground(ctx, canvas.width, canvas.height, backgroundColor);
  drawGrid(ctx, camera);
  initializeViewTransform(ctx, camera);
  const doc = documentManager.document();
  const selection = documentManager.selection();
  drawDocument(ctx, camera, doc, selection);
}
