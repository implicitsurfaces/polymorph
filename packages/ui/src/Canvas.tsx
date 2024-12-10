import { useState, useRef, useCallback, useEffect } from "react";
import { Vector2, Matrix3 } from "threejs-math";
import { Camera2 } from "./Camera2.ts";
import { Point, Layer, Document, ElementId } from "./Document.ts";
import { DocumentManager } from "./DocumentManager.ts";

import "./Canvas.css";

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

function drawGrid(ctx: CanvasRenderingContext2D, camera: Camera2) {
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

function drawDisk(
  ctx: CanvasRenderingContext2D,
  position: Vector2,
  radius: number,
  fillStyle: FillStyle,
) {
  ctx.beginPath();
  ctx.arc(position.x, position.y, radius, 0, 2 * Math.PI);
  ctx.fillStyle = fillStyle;
  ctx.fill();
}

const _pointRadius = 5;

function drawPoint(
  ctx: CanvasRenderingContext2D,
  point: Point,
  isHighlighted: boolean,
) {
  const fillStyle = isHighlighted ? "#4063d5" : "black";
  drawDisk(ctx, point.position, _pointRadius, fillStyle);
}

function drawLayer(
  ctx: CanvasRenderingContext2D,
  document: Document,
  layer: Layer,
  highlightedId: ElementId | undefined,
) {
  layer.points.forEach((id: ElementId) => {
    const point = document.getElementFromId<Point>(id);
    if (point) {
      const isHighlighted = id === highlightedId;
      drawPoint(ctx, point, isHighlighted);
    }
  });
}

function drawDocument(
  ctx: CanvasRenderingContext2D,
  document: Document,
  highlightedId: ElementId | undefined,
) {
  document.layers.forEach((id: ElementId) => {
    const layer = document.getElementFromId<Layer>(id);
    if (layer) {
      drawLayer(ctx, document, layer, highlightedId);
    }
  });
}

function draw(
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
  const highlightedId = documentManager.highlightedElementId();
  drawDocument(ctx, document, highlightedId);
}

///////////////////////////////////////////////////////////////////////////////
//                            Highlight util

interface ClosestElement {
  id: ElementId | undefined;
  distance: number;
}

function findClosestElementInLayer(
  document: Document,
  layer: Layer,
  position: Vector2,
): ClosestElement {
  // Compute distance squared to closest point center
  let closestDistanceSquared = Infinity;
  let closestPoint: Point | undefined = undefined;
  for (const id of layer.points) {
    const point = document.getElementFromId<Point>(id);
    if (point) {
      const d = point.position.distanceToSquared(position);
      if (d < closestDistanceSquared) {
        closestDistanceSquared = d;
        closestPoint = point;
      }
    }
  }

  // Compute distance to closest point disk
  let closestId: ElementId | undefined = undefined;
  let closestDistance = Infinity;
  if (closestPoint) {
    closestId = closestPoint.id;
    if (closestDistanceSquared <= _pointRadius * _pointRadius) {
      // Position is inside the disk
      closestDistance = 0;
    } else {
      // Position is outside the disk
      closestDistance = Math.sqrt(closestDistanceSquared) - _pointRadius;
    }
  }
  return { id: closestId, distance: closestDistance };
}

function findClosestElementInDocument(
  document: Document,
  position: Vector2,
): ClosestElement {
  let closestDistance = Infinity;
  let closestId: ElementId | undefined = undefined;
  for (const id of document.layers) {
    const layer = document.getElementFromId<Layer>(id);
    if (layer) {
      const ce = findClosestElementInLayer(document, layer, position);
      if (ce.distance < closestDistance) {
        closestDistance = ce.distance;
        closestId = ce.id;
      }
    }
  }
  return { id: closestId, distance: closestDistance };
}

function highlight(
  documentManager: DocumentManager,
  mousePosition: Vector2,
  tolerance: number,
) {
  const ce = findClosestElementInDocument(
    documentManager.document(),
    mousePosition,
  );
  if (ce.id && ce.distance < tolerance) {
    documentManager.setHighlightedElement(ce.id);
  } else {
    documentManager.setHighlightedElement(undefined);
  }
}

///////////////////////////////////////////////////////////////////////////////
//                            Position util

/**
 * Returns the offset, in CSS pixels, between the border box and the content box
 * of the given HTML element.
 */
// Note: we need this because while ResizeObserver provides the size of the
// content box, it does not provide its position in any coordinate systems,
// and its position can change even without its size changing (e.g., if the
// user scolls the page).
//
// Therefore, when a pointer event is triggered, the only way to reliably
// access the canvas position in viewport coordinate is to query its border
// box position via getBoundingClientRect(), and substracts the
// padding/border using getComputedStyle(). It might be possible to keep the
// latter cached, but it's probably not worth it.
//
function getBorderBoxToContentBoxOffset(element: HTMLElement): Vector2 {
  const cs = getComputedStyle(element);
  const paddingLeft = parseFloat(cs.paddingLeft);
  const paddingTop = parseFloat(cs.paddingTop);
  const borderLeft = parseFloat(cs.borderLeft);
  const borderTop = parseFloat(cs.borderTop);
  return new Vector2(paddingLeft + borderLeft, paddingTop + borderTop);
}

/**
 * Returns the position of the topleft corner of the content box of the given
 * HTML element (that is, exluding border and padding), in CSS pixels,
 * relative to the topleft corner of the browser's window
 * (= "viewport coordinates").
 */
function getContentBoxPosition(element: HTMLElement): Vector2 {
  const borderBox = element.getBoundingClientRect();
  const offset = getBorderBoxToContentBoxOffset(element);
  return new Vector2(borderBox.left, borderBox.top).add(offset);
}

/**
 * Return the position of the pointer event, in CSS pixels, relative to the
 * topleft corner of the browser's window (= "viewport coordinates").
 */
function getMouseWindowPosition(event: IMouseEvent): Vector2 {
  return new Vector2(event.clientX, event.clientY);
}

/**
 * Return the position of the pointer event, in hardware pixels, relative to
 * the topleft corner of the content box of the given HTML element (that is,
 * exluding border and padding).
 */
function getMouseViewPosition(
  event: IMouseEvent,
  element: HTMLElement,
): Vector2 {
  return getMouseWindowPosition(event) //
    .sub(getContentBoxPosition(element))
    .multiplyScalar(window.devicePixelRatio);
}

/**
 * Return the position of the pointer event in document coordinates, assuming the
 * document is drawn into the content box of the given HTML element using the
 * given camera.
 */
function getMouseDocumentPosition(
  event: IMouseEvent,
  element: HTMLElement,
  camera: Camera2,
) {
  const viewToDocument = camera.viewMatrix().invert();
  return getMouseViewPosition(event, element).applyMatrix3(viewToDocument);
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
  documentManager: DocumentManager;
}

type IMouseEvent = MouseEvent | React.MouseEvent;
type IPointerEvent = PointerEvent | React.PointerEvent;
type IWheelEvent = WheelEvent | React.WheelEvent;

export function Canvas({ documentManager }: CanvasProps) {
  const [camera, setCamera] = useState<Camera2>(new Camera2());
  const [pointerState, setPointerState] = useState<PointerState | null>(null);

  const ref = useRef<HTMLCanvasElement | null>(null);

  // Returns whether there is a drag action available for the drag button.
  //
  const onDragStart = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    (_event: IPointerEvent) => {
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
    },
    [pointerState],
  );

  const onDragMove = useCallback(
    (event: IPointerEvent) => {
      const canvas = ref.current;
      if (!canvas) {
        return;
      }
      if (!pointerState) {
        return;
      }
      const deltaPos = getMouseViewPosition(event, canvas).sub(
        pointerState.viewPosOnPress,
      );
      switch (pointerState.button) {
        case 1: {
          // middle drag: pan
          const nextCenter = pointerState.cameraOnPress.center
            .clone()
            .sub(deltaPos);
          const nextCamera = pointerState.cameraOnPress.clone();
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
    },
    [pointerState],
  );

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const onDragEnd = useCallback((_event: IPointerEvent) => {
    // Nothing for now
  }, []);

  const onClick = useCallback(
    (event: IPointerEvent) => {
      const canvas = ref.current;
      if (!canvas) {
        return;
      }
      if (!pointerState) {
        return;
      }
      switch (event.button) {
        case 0: {
          // left click: create point
          const doc = documentManager.document();
          const layer = documentManager.activeLayer();
          if (layer) {
            const pos = getMouseDocumentPosition(
              event,
              canvas,
              pointerState.cameraOnPress,
            );
            const name = `Point ${layer.points.length + 1}`;
            const point = doc.createPoint({ name: name, position: pos });
            layer.points.push(point.id);
            documentManager.setHighlightedElement(point.id);
            documentManager.commitChanges();
          }
          break;
        }
        case 2: {
          // right click: reset rotation
          const anchor = getMouseViewPosition(event, canvas);
          const nextCamera = pointerState.cameraOnPress.clone();
          nextCamera.setRotationAround(anchor, 0);
          setCamera(nextCamera);
          break;
        }
      }
    },
    [pointerState, documentManager],
  );

  const onPointerDown = useCallback(
    (event: IPointerEvent) => {
      const canvas = ref.current;
      if (!canvas) {
        return;
      }
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
        viewPosOnPress: getMouseViewPosition(event, canvas),
        cameraOnPress: camera.clone(),
        isDrag: false,
        isDragAccepted: false,
      });
    },
    [pointerState, camera],
  );

  const onPointerHover = useCallback(
    (event: IPointerEvent) => {
      const canvas = ref.current;
      if (!canvas) {
        return;
      }
      if (pointerState) {
        // Mouse moves with mouse buttons down are handled by onPointerMove
        return;
      }
      // Compute threshold in document coordinates
      const position = getMouseDocumentPosition(event, canvas, camera);
      const toleranceInPx = 3;
      const toleranceInDocCoords = toleranceInPx / camera.zoom;
      highlight(documentManager, position, toleranceInDocCoords);
    },
    [pointerState, documentManager, camera],
  );

  const onPointerMove = useCallback(
    (event: IPointerEvent) => {
      const canvas = ref.current;
      if (!canvas) {
        return;
      }
      if (!pointerState) {
        // Mouse moves without mouse buttons down are handled by onPointerHover
        return;
      }
      // Disambiguate between drag and click actions
      let nextPointerState = null;
      const dragThreshold = 5;
      const deltaPos = getMouseViewPosition(event, canvas).sub(
        pointerState.viewPosOnPress,
      );
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
    },
    [pointerState, onDragStart, onDragMove],
  );

  const onPointerUp = useCallback(
    (event: IPointerEvent) => {
      // Nothing to do unless we're part of pointerdown-pointermove-pointerup sequence
      // and e.button matches the button of that sequence
      if (!(pointerState && pointerState.button === event.button)) {
        return;
      }
      if (pointerState.isDragAccepted) {
        onDragEnd(event);
      } else {
        onClick(event);
      }
      setPointerState(null);
    },
    [pointerState, onDragEnd, onClick],
  );

  const onWheel = useCallback(
    (event: IWheelEvent) => {
      const canvas = ref.current;
      if (!canvas) {
        return;
      }
      // TODO: support all delta modes
      // 0 = pixels (120px for one scroll step)
      // 1 = lines
      // 2 = pages
      if (event.deltaMode != 0) {
        return;
      }
      const anchor = getMouseViewPosition(event, canvas);
      const steps = (-event.deltaY / 120) * window.devicePixelRatio;
      const nextCamera = camera.clone().zoomAt(anchor, steps);
      setCamera(nextCamera);
    },
    [camera],
  );

  // Redraw whenever the component state is updated, which includes a change
  // of the document, of the camera, or of the canvas width/height trigerred via
  // the ResizeObserver.
  //
  const version = documentManager.version();
  useEffect(() => {
    const canvas = ref.current;
    if (canvas && canvas.width > 0 && canvas.height > 0) {
      draw(canvas, camera, documentManager);
    }
  }, [camera, documentManager, version]);

  // Update the camera (and therefore the canvas width/height attributes)
  // based on its computed device pixel size.
  //
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) {
      return;
    }
    const observer = new ResizeObserver((entries) => {
      const entry = entries.find((entry) => entry.target === canvas);
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
    observer.observe(canvas, { box: "device-pixel-content-box" });
    return () => {
      observer.disconnect();
    };
  }, [camera]);

  // Register for document-wide pointer events once drag starts.
  // This allows to keep dragging even after the pointer exits the canvas.
  //
  useEffect(() => {
    if (pointerState) {
      document.addEventListener("pointermove", onPointerMove);
      document.addEventListener("pointerup", onPointerUp);
    } else {
      document.removeEventListener("pointermove", onPointerMove);
      document.removeEventListener("pointerup", onPointerUp);
    }
    return () => {
      document.removeEventListener("pointermove", onPointerMove);
      document.removeEventListener("pointerup", onPointerUp);
    };
  }, [pointerState, onPointerMove, onPointerUp]);

  return (
    <canvas
      ref={ref}
      width={camera.canvasSize.x}
      height={camera.canvasSize.y}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerHover}
      onWheel={onWheel}
      onContextMenu={(e) => e.preventDefault()}
    />
  );
}

export default Canvas;
