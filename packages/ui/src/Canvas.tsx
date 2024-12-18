import { useState, useRef, useCallback, useEffect } from "react";
import { Vector2 } from "threejs-math";

import { Camera2 } from "./canvas/Camera2.ts";
import { draw } from "./canvas/draw.ts";
import { hover } from "./canvas/hover.ts";
import { useMover } from "./canvas/move.ts";
import {
  getMouseViewPosition,
  getMouseDocumentPosition,
} from "./canvas/getMousePosition.ts";

import { Point, Layer } from "./Document.ts";
import { DocumentManager } from "./DocumentManager.ts";

import "./Canvas.css";

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

type IPointerEvent = PointerEvent | React.PointerEvent;
type IWheelEvent = WheelEvent | React.WheelEvent;

export function Canvas({ documentManager }: CanvasProps) {
  const [camera, setCamera] = useState<Camera2>(new Camera2());
  const [pointerState, setPointerState] = useState<PointerState | null>(null);

  const ref = useRef<HTMLCanvasElement | null>(null);

  const mover = useMover(documentManager);

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
          return mover.start();
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
    [pointerState, mover],
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
      const camera = pointerState.cameraOnPress;
      const vp = pointerState.viewPosOnPress;
      const dp = pointerState.documentPosOnPress;
      const viewDelta = getMouseViewPosition(event, canvas).sub(vp);
      const docDelta = getMouseDocumentPosition(event, canvas, camera).sub(dp);
      switch (pointerState.button) {
        case 0: {
          // left drag: move elements
          mover.move(docDelta);
          break;
        }
        case 1: {
          // middle drag: pan
          const nextCenter = pointerState.cameraOnPress.center
            .clone()
            .sub(viewDelta);
          const nextCamera = pointerState.cameraOnPress.clone();
          nextCamera.center = nextCenter;
          setCamera(nextCamera);
          break;
        }
        case 2: {
          // right drag: rotate
          const rotateSensitivity = 0.01; // 100px -> 1rad
          const anchor = pointerState.viewPosOnPress;
          const angle = rotateSensitivity * (viewDelta.x - viewDelta.y);
          const nextCamera = pointerState.cameraOnPress.clone();
          nextCamera.rotateAround(anchor, angle);
          setCamera(nextCamera);
          break;
        }
      }
    },
    [pointerState, mover],
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
          mover.end();
          break;
        }
      }
    },
    [pointerState, mover],
  );

  const onClick = useCallback(
    (event: IPointerEvent) => {
      const canvas = ref.current;
      const doc = documentManager.document();
      const selection = documentManager.selection();
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
          const hovered = selection.hovered();
          if (hovered) {
            // select
            if (event.shiftKey) {
              selection.toggleSelected(hovered);
            } else {
              selection.setSelected([hovered]);
            }
          } else {
            // create point
            const layer = doc.getElementFromId<Layer>(selection.activeLayer());
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
              selection.setHoveredElement(point.id);
              selection.setSelectedElements([point.id]);
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
      hover(documentManager, camera, position, toleranceInDocCoords);
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
