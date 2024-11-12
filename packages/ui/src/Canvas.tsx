import { useState, useRef, useEffect } from 'react';
import { Vector2 } from 'threejs-math';
import { Camera2 } from './Camera2.ts';
import { Point, Scene, SceneManager } from './Scene.ts';

import './Canvas.css';

///////////////////////////////////////////////////////////////////////////////
//                            Draw util

// A StrokeStyle or FillStyle is:
// - A string parsed as CSS <color> value.
// - A CanvasGradient object (a linear or radial gradient).
// - A CanvasPattern object (a repeating image).
//
// https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/fillStyle
// https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/strokeStyle
//
type FillStyle = string | CanvasGradient | CanvasPattern;
type StrokeStyle = string | CanvasGradient | CanvasPattern;

function drawBackground(ctx: CanvasRenderingContext2D, width: number, height: number, fillStyle: FillStyle) {
  ctx.beginPath();
  ctx.fillStyle = fillStyle;
  ctx.fillRect(0, 0, width, height);
}

function initializeViewTransform(ctx: CanvasRenderingContext2D, camera: Camera2) {
  const e = camera.viewMatrix().elements;
  ctx.setTransform(e[0], e[1], e[3], e[4], e[6], e[7]);
}

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
function drawGridLine(ctx: CanvasRenderingContext2D, sceneToView: Camera2, p1: Vector2, p2: Vector2) {
  p1.applyMatrix3(sceneToView);
  p2.applyMatrix3(sceneToView);
  pixelSnap(p1, p2);
  moveTo(ctx, p1);
  lineTo(ctx, p2);
}

function drawGridCells(
  ctx: CanvasRenderingContext2D,
  sceneToView: Camera2,
  xMin: number,
  xMax: number,
  yMin: number,
  yMax: number,
  size: number,
  strokeStyle: StrokeStyle
) {
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
  ctx.strokeStyle = strokeStyle;
  ctx.stroke();
}

function drawGridAxes(
  ctx: CanvasRenderingContext2D,
  sceneToView: Camera2,
  xMin: number,
  xMax: number,
  yMin: number,
  yMax: number,
  strokeStyle: StrokeStyle
) {
  ctx.beginPath();
  drawGridLine(ctx, sceneToView, new Vector2(xMin, 0), new Vector2(xMax, 0));
  drawGridLine(ctx, sceneToView, new Vector2(0, yMin), new Vector2(0, yMax));
  ctx.lineWidth = 1;
  ctx.strokeStyle = strokeStyle;
  ctx.stroke();
}

function drawGrid(ctx: CanvasRenderingContext2D, camera: Camera2) {
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

function drawDisk(ctx: CanvasRenderingContext2D, position: Vector2, radius: number, fillStyle: FillStyle) {
  ctx.beginPath();
  ctx.arc(position.x, position.y, radius, 0, 2 * Math.PI);
  ctx.fillStyle = fillStyle;
  ctx.fill();
}

function drawPoint(ctx: CanvasRenderingContext2D, point: Point) {
  const radius = 5;
  const fillStyle = 'black';
  drawDisk(ctx, point.position, radius, fillStyle);
}

function drawPoints(ctx: CanvasRenderingContext2D, scene: Scene) {
  scene.points.forEach(point => {
    drawPoint(ctx, point);
  });
}

function drawScene(ctx: CanvasRenderingContext2D, scene: Scene) {
  drawPoints(ctx, scene);
}

function draw(canvas: HTMLCanvasElement, camera: Camera2, scene: Scene) {
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    return;
  }
  ctx.resetTransform();
  drawBackground(ctx, canvas.width, canvas.height, '#e0e0e0');
  drawGrid(ctx, camera);
  initializeViewTransform(ctx, camera);
  drawScene(ctx, scene);
}

///////////////////////////////////////////////////////////////////////////////
//                           Canvas React Component

/**
 * Stores information for handling a pointerdown-pointermove-pointerup
 * sequence on a Canvas.
 */
interface PointerState {
  button: number;
  viewPosOnPress: Vector2;
  cameraOnPress: Camera2;
  isDrag: boolean;
  isDragAccepted: boolean;
}

interface CanvasProps {
  sceneManager: SceneManager;
}

type IMouseEvent = MouseEvent | React.MouseEvent;
type IPointerEvent = PointerEvent | React.PointerEvent;
type IWheelEvent = WheelEvent | React.WheelEvent;

export function Canvas({ sceneManager }: CanvasProps) {
  const [camera, setCamera] = useState<Camera2>(new Camera2());
  const [pointerState, setPointerState] = useState<PointerState | null>(null);

  const ref = useRef<HTMLCanvasElement | null>(null);

  const scene = sceneManager.scene();

  /**
   * We need this because while ResizeObserver provides the size of the
   * content box, it does not provide its position in any coordinate systems,
   * and its position can change even without its size changing (e.g., if the
   * user scolls the page).
   *
   * Therefore, when a pointer event is triggered, the only way to reliably
   * access the canvas position in viewport coordinate is to query its border
   * box position via getBoundingClientRect(), and substracts the
   * padding/border using getComputedStyle(). It might be possible to keep
   * the latter cached, but it's probably not worth it.
   */
  function getCanvasBorderBoxToContentBoxOffset() {
    const canvas = ref.current;
    if (!canvas) {
      return new Vector2(0, 0);
    }
    const cs = getComputedStyle(canvas);
    const paddingLeft = parseFloat(cs.paddingLeft);
    const paddingTop = parseFloat(cs.paddingTop);
    const borderLeft = parseFloat(cs.borderLeft);
    const borderTop = parseFloat(cs.borderTop);
    return new Vector2(paddingLeft + borderLeft, paddingTop + borderTop);
  }

  /**
   * Return the position of the topleft corner of the canvas(exluding border
   * and padding), in CSS pixels, relative to the topleft corner of the
   * browser windows (= "viewport coordinates").
   */
  function getCanvasPosition() {
    const canvas = ref.current;
    if (!canvas) {
      return new Vector2(0, 0);
    }
    const borderBox = canvas.getBoundingClientRect();
    const offset = getCanvasBorderBoxToContentBoxOffset();
    return new Vector2(borderBox.left, borderBox.top).add(offset);
  }

  /**
   * Return the position of the pointer event, in CSS pixels, relative to the
   * topleft corner of the browser windows (= "viewport coordinates").
   */
  function getPointerWindowPosition(event: IMouseEvent) {
    return new Vector2(event.clientX, event.clientY);
  }

  /**
   * Return the position of the pointer event, in hardware pixels, relative to the
   * topleft corner of the canvas (exluding border and padding).
   */
  function getPointerViewPosition(event: IMouseEvent) {
    return getPointerWindowPosition(event) //
      .sub(getCanvasPosition())
      .multiplyScalar(window.devicePixelRatio);
  }

  /**
   * Return the position of the pointer event in scene coordinates.
   */
  function getEventScenePosition(event: IMouseEvent) {
    const viewToScene = camera.viewMatrix().invert();
    return getPointerViewPosition(event).applyMatrix3(viewToScene);
  }

  function onPointerDown(event: IPointerEvent) {
    // Ignore if for some reason e.button is null or undefined
    if (event.button == null) {
      return;
    }
    // Prevent concurrent pointerdown-pointermove-pointerup sequences
    if (pointerState) {
      return;
    }
    setPointerState({
      button: event.button,
      viewPosOnPress: getPointerViewPosition(event),
      cameraOnPress: camera.clone(),
      isDrag: false,
      isDragAccepted: false,
    });
  }

  function onPointerMove(event: IPointerEvent) {
    // Nothing to do unless we're part of pointerdown-pointermove-pointerup sequence
    if (!pointerState) {
      return;
    }
    // Disambiguate between drag and click actions
    let nextPointerState = null;
    const dragThreshold = 5;
    const deltaPos = getPointerViewPosition(event).sub(pointerState.viewPosOnPress);
    let isDragAccepted = pointerState.isDragAccepted;
    if (!pointerState.isDrag && deltaPos.manhattanLength() > dragThreshold) {
      nextPointerState = { ...pointerState };
      nextPointerState.isDrag = true;
      nextPointerState.isDragAccepted = onDragStart(event);
      isDragAccepted = nextPointerState.isDragAccepted;
    }
    if (isDragAccepted) {
      onDragMove(event);
    }
    if (nextPointerState) {
      setPointerState(nextPointerState);
    }
  }

  function onPointerUp(event: IPointerEvent) {
    // Nothing to do unless we're part of pointerdown-pointermove-pointerup sequence
    // and e.button matches the button of that sequence
    if (!(pointerState && pointerState.button === event.button)) {
      return;
    }
    if (pointerState.isDragAccepted) {
      onDragEnd(/* event */);
    } else {
      onClick(event);
    }
    setPointerState(null);
  }

  // Returns whether there is a drag action available for the drag button.
  //
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  function onDragStart(_event: IPointerEvent): boolean {
    if (!pointerState) {
      return false;
    }
    switch (pointerState.button) {
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

  function onDragMove(event: IPointerEvent): void {
    if (!pointerState) {
      return;
    }
    const deltaPos = getPointerViewPosition(event).sub(pointerState.viewPosOnPress);
    switch (pointerState.button) {
      case 1: {
        // middle drag: pan
        const nextCenter = pointerState.cameraOnPress.center.clone().sub(deltaPos);
        const nextCamera = camera.clone();
        nextCamera.center = nextCenter;
        setCamera(nextCamera);
        break;
      }
      case 2: {
        // right drag: rotate
        const rotateSensitivity = 0.01; // 100px -> 1rad
        const anchor = pointerState.viewPosOnPress;
        const angle = rotateSensitivity * (deltaPos.x - deltaPos.y);
        const nextCamera = pointerState.cameraOnPress.clone();
        nextCamera.rotateAround(anchor, angle);
        setCamera(nextCamera);
        break;
      }
    }
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  function onDragEnd(_event: IPointerEvent) {
    // Nothing for now
  }

  function onClick(event: IPointerEvent) {
    if (!pointerState) {
      return;
    }
    switch (event.button) {
      case 0: {
        // left click: create point
        const pos = getEventScenePosition(event);
        scene.addPoint(pos);
        sceneManager.commitChanges();
        break;
      }
      case 2: {
        // right click: reset rotation
        const anchor = getPointerViewPosition(event);
        const nextCamera = pointerState.cameraOnPress.clone();
        nextCamera.setRotationAround(anchor, 0);
        setCamera(nextCamera);
        break;
      }
    }
  }

  function onWheel(event: IWheelEvent) {
    // TODO: support all delta modes
    // 0 = pixels (120px for one scroll step)
    // 1 = lines
    // 2 = pages
    if (event.deltaMode != 0) {
      return;
    }
    const anchor = getPointerViewPosition(event);
    const steps = (-event.deltaY / 120) * window.devicePixelRatio;
    const nextCamera = camera.clone().zoomAt(anchor, steps);
    setCamera(nextCamera);
  }

  // Redraw whenever the component state is updated, which includes a change
  // of the scene, of the camera, or of the canvas width/height trigerred via
  // the ResizeObserver.
  //
  useEffect(() => {
    const canvas = ref.current;
    if (canvas && canvas.width > 0 && canvas.height > 0) {
      draw(canvas, camera, scene);
    }
  });

  // Update the camera (and therefore the canvas width/height attributes)
  // based on its computed device pixel size.
  //
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) {
      return;
    }
    const observer = new ResizeObserver(entries => {
      const entry = entries.find(entry => entry.target === canvas);
      if (entry) {
        const w = entry.devicePixelContentBoxSize[0].inlineSize;
        const h = entry.devicePixelContentBoxSize[0].blockSize;
        if (camera.canvasSize.x != w || camera.canvasSize.y != h) {
          const nextCamera = camera.clone();
          nextCamera.canvasSize = new Vector2(w, h);
          setCamera(nextCamera);
        }
      }
    });
    observer.observe(canvas, { box: 'device-pixel-content-box' });
    return () => {
      observer.disconnect();
    };
  }, [camera]);

  // Register for document-wide pointer events once drag starts.
  // This allows to keep dragging even after the pointer exits the canvas.
  //
  useEffect(() => {
    if (pointerState) {
      document.addEventListener('pointermove', onPointerMove);
      document.addEventListener('pointerup', onPointerUp);
    } else {
      document.removeEventListener('pointermove', onPointerMove);
      document.removeEventListener('pointerup', onPointerUp);
    }
    return () => {
      document.removeEventListener('pointermove', onPointerMove);
      document.removeEventListener('pointerup', onPointerUp);
    };
  }, [pointerState]);

  return (
    <div className="canvas-container">
      <canvas
        ref={ref}
        width={camera.canvasSize.x}
        height={camera.canvasSize.y}
        onPointerDown={onPointerDown}
        onWheel={onWheel}
        onContextMenu={e => e.preventDefault()}
      />
    </div>
  );
}

export default Canvas;
