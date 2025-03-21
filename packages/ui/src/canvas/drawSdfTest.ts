import drawAPI from "../sketch/api";

// For now, drawAPI only supports square image renders
const renderSize = 200;

// For now, we use a quick-and-dirty global variable to check that it works.
// Later, we want to sync this with the Document and/or Selection state.
//
const cacheImage: {
  current: null | Uint8ClampedArray;
} = { current: null };

drawAPI.render(renderSize).then((res: Uint8ClampedArray) => {
  cacheImage.current = res;
});

export function drawSdfTest(canvas: HTMLCanvasElement) {
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }
  if (cacheImage.current) {
    const dx = 0;
    const dy = 0;
    const data = new ImageData(cacheImage.current, renderSize, renderSize);
    ctx.putImageData(data, dx, dy);
  }
}
