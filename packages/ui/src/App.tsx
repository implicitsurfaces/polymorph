import { useEffect } from 'react';
import './App.css';

// TODO: move different functionality to different files
// For now it's quick-and-dirty as a proof of concept.

/**
 * Class representing a 2D point.
 */
// TODO: Use Point from sketch package?
//
class Point {
  constructor(
    public x: number = 0,
    public y: number = 0
  ) {}
}

/**
 * Global list storing all the points in the scene.
 */
// TODO: Proper Scene class or similar.
//
const points: Array<Point> = [];

/**
 * Returns the main canvas element.
 */
// TODO: Avoid having a global canvas element: we may
// want to have more than one canvas.
//
function getGlobalCanvas() {
  return document.getElementById('canvas');
}

/**
 * Draws a disk to the given canvas context.
 */
// TODO: move this (and other similar functions) directly
// as methods of the Canvas react element?
//
function drawDisk(ctx, position: Point, radius: number) {
  ctx.beginPath();
  ctx.arc(position.x, position.y, radius, 0, 2 * Math.PI);
  ctx.fillStyle = 'black';
  ctx.fill();

  // For reference: here is how to stroke it:
  // ctx.lineWidth = 4;
  // ctx.strokeStyle = "blue";
  // ctx.stroke();
}

function draw(canvas) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const radius = 5;
  points.forEach(p => {
    drawDisk(ctx, p, radius);
  });
}

function addPoint(e) {
  const canvas = getGlobalCanvas();
  points.push(new Point(e.clientX, e.clientY));
  draw(canvas);
}

/**
 * Sets the size of the canvas' render target (in pixels) to be equal to its
 * display size as an HTML element (in CSS units).
 *
 * This is required since it is not done automatically, and therefore we would
 * by default get a small render target (e.g., 100x100 px) whose pixels are
 * stretched to fill the size of the HTML element.
 */
function updateCanvasSize(canvas) {
  canvas.width = canvas.clientWidth;
  canvas.height = canvas.clientHeight;
  // TODO: With hi-res screens, shouldn't we instead use the ratio between CSS
  // units and physical pixels? Also, currently, if the browser has a zoom factor,
  // this also leads to pixelization.
}

function updateCanvas(canvas) {
  updateCanvasSize(canvas);
  draw(canvas);
}

// Update canvas on resize
//
// TODO: Reactify this? Should be possible with a ResizeObserver hook
// https://developer.mozilla.org/en-US/docs/Web/API/ResizeObserver
// https://react.dev/reference/react/hooks
// https://blog.logrocket.com/using-resizeobserver-react-responsive-designs/
//
window.addEventListener('resize', () => {
  const canvas = getGlobalCanvas();
  updateCanvas(canvas);
});

export function Canvas() {
  // Call updateCanvasSize when the component is first loaded
  useEffect(() => {
    // TODO: how to update current canvas instead of global?
    updateCanvasSize(getGlobalCanvas());
  });

  // TODO: how to pass the current canvas to the addPoint callback?
  return <canvas id="canvas" onClick={e => addPoint(e)} />;
}

function App() {
  return (
    <>
      <Canvas />
    </>
  );
}

export default App;
