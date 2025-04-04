import drawAPI from "../sketch/api";
import { Document } from "../doc/Document";
import { DocumentManager } from "../doc/DocumentManager";
import { proxy } from "comlink";

// For now, drawAPI only supports square image renders
const renderSize = 200;

// For now, we use a quick-and-dirty global variable to check that it works.
// Later, we want to sync this with the Document and/or Selection state.
//
const cacheImage: {
  current: null | Uint8ClampedArray;
} = { current: null };

export function updateSdfTest(doc: Document) {
  console.log("updateSdfTest", doc);
  drawAPI.render(proxy(doc), renderSize).then((res: Uint8ClampedArray) => {
    cacheImage.current = res;
  });
}

export function drawSdfTest(
  canvas: HTMLCanvasElement,
  documentManager: DocumentManager,
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }
  const doc = documentManager.document();

  updateSdfTest(doc);

  // TODO: redraw when the asynchronous computation cacheImage.current finishes.
  if (cacheImage.current) {
    const dx = 0;
    const dy = 0;
    const data = new ImageData(cacheImage.current, renderSize, renderSize);
    ctx.putImageData(data, dx, dy);
  }
}
