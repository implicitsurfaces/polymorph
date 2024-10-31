import { useEffect } from 'react';
import { Vector2 } from 'threejs-math';

import { Camera2 } from './Camera2.ts';
import { getActiveScene } from './Scene.ts';

import './Canvas.css';

///////////////////////////////////////////////////////////////////////////////
//                            Draw util

function drawBackground(ctx, width, height, color) {
  ctx.resetTransform();
  ctx.beginPath();
  ctx.fillStyle = color;
  ctx.fillRect(0, 0, width, height);
}

function initializeViewTransform(ctx, camera) {
  const e = camera.viewMatrix().elements;
  ctx.setTransform(e[0], e[1], e[3], e[4], e[6], e[7]);
}

function drawGrid(ctx) {
  // For now, just draw the x-axis and y-axis unclipped
  // in world coords for testing. Because of antialiasing
  // this looks ugly. We want to change this to draw in
  // view coords and with pixel rounding so that we get
  // pixel perfect 1px-wide lines.
  ctx.beginPath();
  ctx.moveTo(-10000, 0);
  ctx.lineTo(10000, 0);
  ctx.moveTo(0, -10000);
  ctx.lineTo(0, 10000);
  ctx.stroke();
}

function drawDisk(ctx, position: Point, radius: number) {
  ctx.beginPath();
  ctx.arc(position.x, position.y, radius, 0, 2 * Math.PI);
  ctx.fillStyle = 'black';
  ctx.fill();
}

function drawPoints(ctx, scene) {
  const radius = 5;
  scene.points.forEach(p => {
    drawDisk(ctx, p, radius);
  });
}

function drawScene(ctx, scene) {
  drawPoints(ctx, scene);
}

function draw(canvas, camera, scene) {
  const ctx = canvas.getContext('2d');
  drawBackground(ctx, canvas.width, canvas.height, '#e0e0e0');
  initializeViewTransform(ctx, camera);
  drawGrid(ctx);
  drawScene(ctx, scene);
}

///////////////////////////////////////////////////////////////////////////////
//                       Canvas and Camera management

// For now we use a global canvas and camera.
// In the future we should allow multiple canvases obviously (top view vs. 3D view, etc.)

function getActiveCanvas() {
  return document.getElementById('canvas');
}

const _globalCamera = new Camera2();

function getActiveCamera() {
  return _globalCamera;
}

/**
 * Sets the size of the canvas' render target (in pixels) to be equal to its
 * display size as an HTML element (in CSS units).
 *
 * This is required since it is not done automatically, and therefore we would
 * by default get a small render target (e.g., 100x100 px) whose pixels are
 * stretched to fill the size of the HTML element.
 */
function updateCanvasSize(canvas, camera) {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  canvas.width = w;
  canvas.height = h;
  camera.canvasSize = new Vector2(w, h);
  // TODO: With hi-res screens, shouldn't we instead use the ratio between CSS
  // units and physical pixels? Also, currently, if the browser has a zoom factor,
  // this also leads to pixelization.
}

function updateCanvasSizeAndRedraw(canvas, camera, scene) {
  updateCanvasSize(canvas, camera);
  draw(canvas, camera, scene);
}

function updateActiveCanvasSizeAndRedraw() {
  updateCanvasSizeAndRedraw(getActiveCanvas(), getActiveCamera(), getActiveScene());
}

function redrawActiveCanvas() {
  draw(getActiveCanvas(), getActiveCamera(), getActiveScene());
}

///////////////////////////////////////////////////////////////////////////////
//                           Event util

// TODO: add canvas argument or refactor into the Canvas class, and make the
// camera a data member of the Canvas class.

function getEventPosition(e) {
  return new Vector2(e.clientX, e.clientY);
}

function getEventWorldPosition(e) {
  const camera = getActiveCamera();
  const viewToWorld = camera.viewMatrix().invert();
  return getEventPosition(e).applyMatrix3(viewToWorld);
}

///////////////////////////////////////////////////////////////////////////////
//                           Mouse events

// TODO: add an observer system so that modifying the camera or the scene
// automatically causes a redraw. For now we manually call redraws.

// Store mouse state.
// TODO: avoid globals by storing as state of the Canvas component?
let _currentMouseButton = null;
let _mousePosOnPress = null;
let _cameraOnPress = null;

function onMouseDown(e) {
  // Prevent concurrent mouse actions
  if (_currentMouseButton != null && _currentMouseButton != e.button) {
    return;
  }

  _currentMouseButton = e.button;
  _mousePosOnPress = getEventPosition(e);
  _cameraOnPress = getActiveCamera().clone();
}

function onMouseMove(e) {
  if (_currentMouseButton == null) {
    return;
  }

  switch (_currentMouseButton) {
    case 1: {
      // middle button: pan
      const delta = getEventPosition(e);
      delta.sub(_mousePosOnPress);
      const newCenter = _cameraOnPress.center.clone();
      newCenter.sub(delta);
      getActiveCamera().center = newCenter;
      redrawActiveCanvas();
      break;
    }
    case 2: {
      // right button: rotate (TODO)
      break;
    }
  }
}

function onMouseUp(e) {
  if (_currentMouseButton != e.button) {
    return;
  }
  switch (e.button) {
    case 0: {
      // left button: create point
      const pos = getEventWorldPosition(e);
      getActiveScene().addPoint(pos);
      redrawActiveCanvas();
      break;
    }
  }
  _currentMouseButton = null;
}

function onWheel(e) {
  // TODO: support all delta modes
  // 0 = pixels (120px for one scroll step)
  // 1 = lines
  // 2 = pages
  if (e.deltaMode != 0) {
    return;
  }

  const anchor = getEventPosition(e);
  const steps = -e.deltaY / 120;
  getActiveCamera().zoomAt(anchor, steps);
  redrawActiveCanvas();
}

///////////////////////////////////////////////////////////////////////////////
//                           Canvas React Component

// TODO: avoid globals by passing the relevant canvas to the callbacks.

// Update canvas on resize
//
// TODO: Reactify this? Should be possible with a ResizeObserver hook
// https://developer.mozilla.org/en-US/docs/Web/API/ResizeObserver
// https://react.dev/reference/react/hooks
// https://blog.logrocket.com/using-resizeobserver-react-responsive-designs/
//
window.addEventListener('resize', () => {
  updateActiveCanvasSizeAndRedraw();
});

export function Canvas() {
  // update canvas on first-time load
  useEffect(() => {
    updateActiveCanvasSizeAndRedraw();
  });

  return (
    <canvas
      id="canvas"
      onMouseDown={e => onMouseDown(e)}
      onMouseMove={e => onMouseMove(e)}
      onMouseUp={e => onMouseUp(e)}
      onWheel={e => onWheel(e)}
    />
  );
}
