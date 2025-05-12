import drawAPI from "../sketch/api";
import { Document } from "../doc/Document";
import { DocumentManager } from "../doc/DocumentManager";

// For now, drawAPI only supports square image renders
const renderSize = 512;

class RenderQueue {
  private _jsonQueue: string[] = [];
  private _lastRender: Uint8ClampedArray | undefined;

  constructor(readonly canvas: HTMLCanvasElement) {}

  private _startRender() {
    if (this._jsonQueue.length >= 1) {
      drawAPI
        .render(this._jsonQueue[0], renderSize)
        .then((res: Uint8ClampedArray) => {
          this._lastRender = res;
          this._jsonQueue.splice(0, 1);
          this._startRender();
          this._updateCanvas();
        });
    }
  }

  private _updateCanvas() {
    const lastRender = this.lastRender();
    if (!lastRender) {
      return;
    }
    const ctx = this.canvas.getContext("2d");
    if (!ctx) {
      return;
    }
    const dx = 0;
    const dy = 0;
    const data = new ImageData(lastRender, renderSize, renderSize);
    ctx.setTransform(1, 0, 0, 1, 0, 0); // identity
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.putImageData(data, dx, dy);
  }

  push(doc: Document) {
    if (this._jsonQueue.length > 1) {
      // For debouncing reason, we only keep in the queue at most two documents:
      // the currently-rendering and the last-pushed doc. So if the queue already
      // has two or more document, we remove them all except the currently-rendering one
      this._jsonQueue.splice(1);
    }
    this._jsonQueue.push(doc.toJSON());
    if (this._jsonQueue.length == 1) {
      this._startRender();
    } else {
      // This means that there is already a render in progress. The
      // just-pushed document will automatically start rendering as soon as
      // the currently-rendering is finished.
    }
  }

  lastRender(): Uint8ClampedArray | undefined {
    return this._lastRender;
  }
}

// For each canvas, we store its render  there is a render in progress, and
// whether other rendersanother
const renderQueues = new Map<HTMLCanvasElement, RenderQueue>();

function getOrCreateRenderQueue(canvas: HTMLCanvasElement) {
  if (!renderQueues.has(canvas)) {
    renderQueues.set(canvas, new RenderQueue(canvas));
  }
  return renderQueues.get(canvas)!;
}

export function drawSdfTest(
  canvas: HTMLCanvasElement,
  documentManager: DocumentManager,
) {
  const doc = documentManager.document();
  const queue = getOrCreateRenderQueue(canvas);
  queue.push(doc);
}
