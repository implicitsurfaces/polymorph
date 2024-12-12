import { useState, useRef, useCallback, useEffect } from "react";
import { Vector2, Matrix3 } from "threejs-math";
import { Camera2 } from "./Camera2.ts";
import {
  Point,
  Layer,
  Document,
  ElementId,
  Element,
  EdgeElement,
} from "./Document.ts";
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

function getPrimaryColor(isHighlighted: boolean, isSelected: boolean): string {
  if (isSelected) {
    return "#4063d5";
  } else if (isHighlighted) {
    return "#96a4d3";
  } else {
    return "black";
  }
}

function drawPoint(
  ctx: CanvasRenderingContext2D,
  point: Point,
  isHighlighted: boolean,
  isSelected: boolean,
) {
  const fillStyle = getPrimaryColor(isHighlighted, isSelected);
  drawDisk(ctx, point.position, _pointRadius, fillStyle);
}

function drawPoints(
  ctx: CanvasRenderingContext2D,
  document: Document,
  elements: Array<ElementId>,
  highlightedId: ElementId | undefined,
  selectedIds: Array<ElementId>,
) {
  for (const id of elements) {
    const element = document.getElementFromId(id);
    if (element && element.type === "Point") {
      const isHighlighted = id === highlightedId;
      const isSelected = selectedIds.includes(id);
      drawPoint(ctx, element, isHighlighted, isSelected);
    }
  }
}

interface CanvasShapeBase {
  type: string;
}

interface CanvasArc extends CanvasShapeBase {
  type: "Arc";
  center: Vector2;
  radius: number;
  startAngle: number;
  endAngle: number;
  isCounterClockwise: boolean;
}

interface CanvasLineSegment extends CanvasShapeBase {
  type: "LineSegment";
  startPoint: Vector2;
  endPoint: Vector2;
}

type CanvasGeneralizedArc = CanvasArc | CanvasLineSegment;
type CanvasShape = CanvasGeneralizedArc;

function drawLineSegment(
  ctx: CanvasRenderingContext2D,
  lineSegment: CanvasLineSegment,
  isHighlighted: boolean,
  isSelected: boolean,
) {
  ctx.beginPath();
  ctx.moveTo(lineSegment.startPoint.x, lineSegment.startPoint.y);
  ctx.lineTo(lineSegment.endPoint.x, lineSegment.endPoint.y);
  ctx.lineWidth = 2;
  ctx.strokeStyle = getPrimaryColor(isHighlighted, isSelected);
  ctx.stroke();
}

function drawArc(
  ctx: CanvasRenderingContext2D,
  arc: CanvasArc,
  isHighlighted: boolean,
  isSelected: boolean,
) {
  ctx.beginPath();
  ctx.arc(
    arc.center.x,
    arc.center.y,
    arc.radius,
    arc.startAngle,
    arc.endAngle,
    arc.isCounterClockwise,
  );
  ctx.lineWidth = 2;
  ctx.strokeStyle = getPrimaryColor(isHighlighted, isSelected);
  ctx.stroke();
}

function drawControlPoint(ctx: CanvasRenderingContext2D, point: Vector2) {
  ctx.beginPath();
  const fillStyle = "#ff0000";
  const diskRadius = 5;
  drawDisk(ctx, point, diskRadius, fillStyle);
}

function drawShape(
  ctx: CanvasRenderingContext2D,
  shape: CanvasShape,
  isHighlighted: boolean,
  isSelected: boolean,
) {
  switch (shape.type) {
    case "Arc":
      drawArc(ctx, shape, isHighlighted, isSelected);
      break;
    case "LineSegment":
      drawLineSegment(ctx, shape, isHighlighted, isSelected);
      break;
  }
}

function getLineSegment(
  startPoint: Vector2,
  endPoint: Vector2,
): CanvasLineSegment {
  return { type: "LineSegment", startPoint: startPoint, endPoint: endPoint };
}

function getGeneralizedArcFromStartTangent(
  startPoint: Vector2,
  endPoint: Vector2,
  tangent: Vector2,
): CanvasGeneralizedArc {
  // Fallback to line segment in degenerate cases (collinear)
  const chord = endPoint.clone().sub(startPoint);
  const s_ = chord.cross(tangent);
  if (s_ === 0) {
    return { type: "LineSegment", startPoint: startPoint, endPoint: endPoint };
  }

  // Compute center
  const tangentLength = tangent.length(); // guaranteed > 0 otherwise s_ == 0
  const s = s_ / tangentLength;
  const c = chord.dot(tangent) / tangentLength;
  const bulge = -s / (c + Math.sqrt(c * c + s * s));
  const chordPerp = new Vector2(-chord.y, chord.x);
  const bb = (bulge - 1 / bulge) / 4;
  const midPoint = startPoint.clone().add(endPoint).multiplyScalar(0.5);
  const center = midPoint.clone().sub(chordPerp.multiplyScalar(bb));

  return {
    type: "Arc",
    center: center,
    radius: center.distanceTo(startPoint),
    startAngle: startPoint.clone().sub(center).angle(),
    endAngle: endPoint.clone().sub(center).angle(),
    isCounterClockwise: s_ > 0,
  };
}

function getCCurveShapes(
  startPoint: Vector2,
  endPoint: Vector2,
  controlPoint: Vector2,
): Array<CanvasGeneralizedArc> {
  const w0 = controlPoint.distanceTo(endPoint);
  const w1 = controlPoint.distanceTo(startPoint);
  const w2 = startPoint.distanceTo(endPoint);

  const junction = startPoint
    .clone()
    .multiplyScalar(w0)
    .add(endPoint.clone().multiplyScalar(w1))
    .add(controlPoint.clone().multiplyScalar(w2))
    .divideScalar(w0 + w1 + w2);

  const startTangent = controlPoint.clone().sub(startPoint);
  const endTangent = controlPoint.clone().sub(endPoint);

  return [
    getGeneralizedArcFromStartTangent(startPoint, junction, startTangent),
    getGeneralizedArcFromStartTangent(endPoint, junction, endTangent),
  ];
}

function getStartAndEndPositions(document: Document, element: EdgeElement) {
  const startPoint = document.getElementFromId<Point>(element.startPoint);
  const endPoint = document.getElementFromId<Point>(element.endPoint);
  if (startPoint && endPoint) {
    return { start: startPoint.position, end: endPoint.position };
  } else {
    return undefined;
  }
}

function drawEdges(
  ctx: CanvasRenderingContext2D,
  document: Document,
  elements: Array<ElementId>,
  highlightedId: ElementId | undefined,
  selectedIds: Array<ElementId>,
) {
  for (const id of elements) {
    const element = document.getElementFromId(id);
    if (element) {
      const shapes: Array<CanvasShape> = [];
      const controlPoints: Array<Vector2> = [];
      switch (element.type) {
        case "LineSegment": {
          const p = getStartAndEndPositions(document, element);
          if (p) {
            shapes.push(getLineSegment(p.start, p.end));
          }
          break;
        }
        case "ArcFromStartTangent": {
          const p = getStartAndEndPositions(document, element);
          if (p) {
            const tangent = element.tangent;
            const controlPoint = tangent.clone().add(p.start);
            shapes.push(
              getGeneralizedArcFromStartTangent(p.start, p.end, tangent),
            );
            controlPoints.push(controlPoint);
          }
          break;
        }
        case "CCurve": {
          const p = getStartAndEndPositions(document, element);
          if (p) {
            let controlPoint = element.controlPoint;
            switch (element.mode) {
              case "startTangent":
                controlPoint = controlPoint.clone().add(p.start);
                break;
              case "endTangent":
                controlPoint = controlPoint.clone().add(p.end);
                break;
            }
            const shapes_ = getCCurveShapes(p.start, p.end, controlPoint);
            for (const shape of shapes_) {
              shapes.push(shape);
            }
            controlPoints.push(controlPoint);
          }
          break;
        }
      }
      const isHighlighted = id === highlightedId;
      const isSelected = selectedIds.includes(id);
      for (const shape of shapes) {
        drawShape(ctx, shape, isHighlighted, isSelected);
      }
      for (const point of controlPoints) {
        drawControlPoint(ctx, point);
      }
      // TODO: draw construction lines, e.g., from startPoint to controlPoint
    }
  }
}

function drawDocument(
  ctx: CanvasRenderingContext2D,
  document: Document,
  highlightedId: ElementId | undefined,
  selectedIds: Array<ElementId>,
) {
  document.layers.forEach((id: ElementId) => {
    const layer = document.getElementFromId<Layer>(id);
    if (layer) {
      // Note: we use two passes since we want to draw all points on top of
      // edges, regardless of layer order.
      drawEdges(ctx, document, layer.elements, highlightedId, selectedIds);
      drawPoints(ctx, document, layer.elements, highlightedId, selectedIds);
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
  const selectedIds = documentManager.selectedElementIds();
  drawDocument(ctx, document, highlightedId, selectedIds);
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
  // For now, we only look for points
  for (const id of layer.elements) {
    const element = document.getElementFromId(id);
    if (element && element.type === "Point") {
      const d = element.position.distanceToSquared(position);
      if (d < closestDistanceSquared) {
        closestDistanceSquared = d;
        closestPoint = element;
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
  documentPosOnPress: Vector2;
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

interface MovedElementInfo {
  element: Element;
  positionOnPress: Vector2;
}

export function Canvas({ documentManager }: CanvasProps) {
  const [camera, setCamera] = useState<Camera2>(new Camera2());
  const [pointerState, setPointerState] = useState<PointerState | null>(null);

  const ref = useRef<HTMLCanvasElement | null>(null);

  const movedElementInfos = useRef<Array<MovedElementInfo>>([]);

  // Returns whether there is a drag action available for the drag button.
  //
  const onDragStart = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    (_event: IPointerEvent) => {
      if (!pointerState) {
        return false;
      }
      switch (pointerState.button) {
        case 0: {
          // left drag: move elements
          const highlightedElement = documentManager.highlightedElement();
          if (!highlightedElement) {
            return false;
          } else {
            const selectedElements = documentManager.selectedElements();
            let movedElements: Array<Element> = [];
            if (selectedElements.includes(highlightedElement)) {
              movedElements = selectedElements;
            } else {
              // Moving an highlighted element that is not currently selected
              // makes it the currently selected element.
              movedElements = [highlightedElement];
              documentManager.setSelectedElements([highlightedElement.id]);
            }
            // Remember start positions of all elements.
            // We only do this if not done already, otherwise
            // in React strict mode it might be done twice, and
            // the second time the point.position might have already
            // been moved a little.
            if (movedElementInfos.current.length == 0) {
              const infos: Array<MovedElementInfo> = [];
              for (const element of movedElements) {
                if (element.type === "Point") {
                  infos.push({
                    element: element,
                    positionOnPress: element.position.clone(),
                  });
                }
              }
              movedElementInfos.current = infos;
            }
            if (movedElementInfos.current.length > 0) {
              return true;
            } else {
              return false;
            }
          }
        }
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
    [pointerState, documentManager],
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
        case 0: {
          // left drag: move elements
          const documentPosOnPress = pointerState.documentPosOnPress;
          const documentPos = getMouseDocumentPosition(
            event,
            canvas,
            pointerState.cameraOnPress,
          );
          const delta = documentPos.sub(documentPosOnPress);
          for (const info of movedElementInfos.current) {
            if (info.element.type == "Point") {
              const point = info.element;
              point.position = info.positionOnPress.clone().add(delta);
            }
          }
          documentManager.stageChanges();
          break;
        }
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
    [pointerState, documentManager],
  );

  const onDragEnd = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    (_event: IPointerEvent) => {
      if (!pointerState) {
        return;
      }
      switch (pointerState.button) {
        case 0: {
          // left drag: move elements
          movedElementInfos.current = []; // See onDragStart
          documentManager.commitChanges();
          break;
        }
      }
    },
    [pointerState, documentManager],
  );

  const onClick = useCallback(
    (event: IPointerEvent) => {
      const canvas = ref.current;
      if (!canvas) {
        return;
      }
      if (!pointerState) {
        return;
      }
      // TODO: more generic dispatch with registry of actions with
      // corresponding mouse buttons and modifiers. For now we do
      // a quick-and-dirty hard-coded dispatch.
      switch (event.button) {
        case 0: {
          // left click: create point or select
          const doc = documentManager.document();
          const highlightedElement = documentManager.highlightedElement();
          if (highlightedElement) {
            // select
            const highlightedId = highlightedElement.id;
            if (event.shiftKey) {
              documentManager.toggleSelectedElement(highlightedId);
            } else {
              documentManager.setSelectedElements([highlightedId]);
            }
          } else {
            // create point
            const layer = documentManager.activeLayer();
            if (layer) {
              const pos = getMouseDocumentPosition(
                event,
                canvas,
                pointerState.cameraOnPress,
              );
              const name = doc.findAvailableName("Point ", layer.elements);
              const point = doc.createElement(Point, {
                name: name,
                position: pos,
              });
              layer.elements.push(point.id);
              documentManager.setHighlightedElement(point.id);
              documentManager.setSelectedElements([point.id]);
              documentManager.commitChanges();
            }
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
        documentPosOnPress: getMouseDocumentPosition(event, canvas, camera),
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
