import { useEffect } from 'react';
import { Vector2 } from 'threejs-math';

import { Camera2 } from './Camera2.ts';
import { getActiveScene } from './Scene.ts';

import './Canvas.css';

///////////////////////////////////////////////////////////////////////////////
//                            Draw util

function drawBackground(ctx, width, height, color) {
  ctx.beginPath();
  ctx.fillStyle = color;
  ctx.fillRect(0, 0, width, height);
}

function initializeViewTransform(ctx, camera) {
  const e = camera.viewMatrix().elements;
  ctx.setTransform(e[0], e[1], e[3], e[4], e[6], e[7]);
}

function moveTo(ctx, p: Vector2) {
  ctx.moveTo(p.x, p.y);
}

function lineTo(ctx, p: Vector2) {
  ctx.lineTo(p.x, p.y);
}

/**
 * If the line segment [p1, p2] is perfectly horizontal or perfectly vertical,
 * then snap it to the pixel grid for pixel-perfect rendering.
 */
function pixelSnap(p1: Vector2, p2: Vector2) {
  if (p1.x == p2.x) {
    p1.round();
    p2.round();
    p1.x -= 0.5;
    p2.x -= 0.5;
  } else if (p1.y == p2.y) {
    p1.round();
    p2.round();
    p1.y -= 0.5;
    p2.y -= 0.5;
  }
}

// Transform p1 and p2 to view coords, pixel snap them, then moveTo(p1) and lineTo(p2)
function drawGridLine(ctx, sceneToView, p1, p2) {
  p1.applyMatrix3(sceneToView);
  p2.applyMatrix3(sceneToView);
  pixelSnap(p1, p2);
  moveTo(ctx, p1);
  lineTo(ctx, p2);
}

function drawGridCells(ctx, sceneToView, xMin, xMax, yMin, yMax, size, color) {
  // Get scene coords of the first visible horizontal and vertical grid line,
  // as well as how many of them are visible.
  const xStart = Math.floor(xMin / size) * size;
  const xNum = Math.floor((xMax - xMin) / size) + 2;
  const yStart = Math.floor(yMin / size) * size;
  const yNum = Math.floor((yMax - yMin) / size) + 2;

  ctx.beginPath();

  // Vertical lines
  for (let i = 0; i < xNum; ++i) {
    const x = xStart + i * size;
    drawGridLine(ctx, sceneToView, new Vector2(x, yMin), new Vector2(x, yMax));
  }

  // Horizontal lines
  for (let i = 0; i < yNum; ++i) {
    const y = yStart + i * size;
    drawGridLine(ctx, sceneToView, new Vector2(xMin, y), new Vector2(xMax, y));
  }

  ctx.lineWidth = 1;
  ctx.strokeStyle = color;
  ctx.stroke();
}

function drawGridAxes(ctx, sceneToView, xMin, xMax, yMin, yMax, color) {
  ctx.beginPath();
  drawGridLine(ctx, sceneToView, new Vector2(xMin, 0), new Vector2(xMax, 0));
  drawGridLine(ctx, sceneToView, new Vector2(0, yMin), new Vector2(0, yMax));
  ctx.lineWidth = 1;
  ctx.strokeStyle = color;
  ctx.stroke();
}

function drawGrid(ctx, camera) {
  const sceneToView = camera.viewMatrix();
  const viewToScene = sceneToView.clone().invert();
  const w = camera.canvasSize.x;
  const h = camera.canvasSize.y;
  const axesColor = '#808080';
  const majorColor = '#b2b2b2';
  const minorColor = '#cccccc';

  // Size of minor and major grid cells in scene coords
  const minorSize = Math.pow(10, Math.ceil(0.6 - Math.log10(camera.zoom)));
  const majorSize = minorSize * 10;

  // Express view rectangle in scene coordinates
  const p1 = new Vector2(0, 0).applyMatrix3(viewToScene);
  const p2 = new Vector2(w, 0).applyMatrix3(viewToScene);
  const p3 = new Vector2(w, h).applyMatrix3(viewToScene);
  const p4 = new Vector2(0, h).applyMatrix3(viewToScene);

  // Get min/max in scene coordinates
  const xMin = Math.min(p1.x, p2.x, p3.x, p4.x);
  const xMax = Math.max(p1.x, p2.x, p3.x, p4.x);
  const yMin = Math.min(p1.y, p2.y, p3.y, p4.y);
  const yMax = Math.max(p1.y, p2.y, p3.y, p4.y);

  drawGridCells(ctx, sceneToView, xMin, xMax, yMin, yMax, minorSize, minorColor);
  drawGridCells(ctx, sceneToView, xMin, xMax, yMin, yMax, majorSize, majorColor);
  drawGridAxes(ctx, sceneToView, xMin, xMax, yMin, yMax, axesColor);
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
  ctx.resetTransform();
  drawBackground(ctx, canvas.width, canvas.height, '#e0e0e0');
  drawGrid(ctx, camera);
  initializeViewTransform(ctx, camera);
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
let _mousePosOnPress = null;
let _cameraOnPress = null;
let _dragButton = null;
let _isDrag = false; // whether mouse moved more than threshold
let _isDragAccepted = false; // whether there is a drag action for drag button

function onMouseDown(e) {
  // Prevent concurrent drag/click actions
  if (_dragButton != null && _dragButton != e.button) {
    return;
  }
  _dragButton = e.button;
  _mousePosOnPress = getEventPosition(e);
  _cameraOnPress = getActiveCamera().clone();
  _isDrag = false;
  _isDragAccepted = false;
}

function onMouseMove(e) {
  // For now, we do nothing on mouse move unless a button is pressed
  if (_dragButton == null) {
    return;
  }

  // Disambiguate between drag and click actions
  const dragThreshold = 5;
  const deltaPos = getEventPosition(e).sub(_mousePosOnPress);
  if (!_isDrag && deltaPos.manhattanLength() > dragThreshold) {
    _isDrag = true;
    _isDragAccepted = onDragStart(e);
  }
  if (_isDragAccepted) {
    onDragMove(e);
  }
}

function onMouseUp(e) {
  if (_dragButton != e.button) {
    return;
  }
  if (_isDragAccepted) {
    onDragEnd(e);
  } else {
    onClick(e);
  }
  _dragButton = null;
}

// Returns whether there is a drag action available for the drag button.
//
function onDragStart(/* e */): boolean {
  switch (_dragButton) {
    case 1: {
      // middle drag: pan
      return true;
    }
    case 2: {
      // right drag: rotate
      return true;
    }
  }
  return false;
}

function onDragMove(e) {
  const deltaPos = getEventPosition(e).sub(_mousePosOnPress);
  switch (_dragButton) {
    case 1: {
      // middle drag: pan
      const newCenter = _cameraOnPress.center.clone().sub(deltaPos);
      getActiveCamera().center = newCenter;
      redrawActiveCanvas();
      break;
    }
    case 2: {
      // right drag: rotate
      const rotateSensitivity = 0.01; // 100px -> 1rad
      const anchor = _mousePosOnPress;
      const angle = rotateSensitivity * (deltaPos.x - deltaPos.y);
      getActiveCamera().copy(_cameraOnPress).rotateAround(anchor, angle);
      redrawActiveCanvas();
      break;
    }
  }
}

function onDragEnd(/* e */) {
  // Nothing for now
}

function onClick(e) {
  switch (e.button) {
    case 0: {
      // left click: create point
      const pos = getEventWorldPosition(e);
      getActiveScene().addPoint(pos);
      redrawActiveCanvas();
      break;
    }
    case 2: {
      // right click: reset rotation
      const anchor = _mousePosOnPress;
      getActiveCamera().setRotationAround(anchor, 0);
      redrawActiveCanvas();
      break;
    }
  }
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
      onContextMenu={e => e.preventDefault()}
    />
  );
}
