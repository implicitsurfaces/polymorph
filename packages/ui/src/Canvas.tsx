import { useState, useRef, useCallback, useEffect, useContext } from "react";
import { Vector2 } from "threejs-math";

import { Camera2 } from "./canvas/Camera2.ts";
import { draw } from "./canvas/draw.ts";
import { CanvasPointerEvent } from "./canvas/events.ts";
import {
  getMouseViewPosition,
  getMouseDocumentPosition,
} from "./canvas/getMousePosition.ts";

import { DocumentManager } from "./DocumentManager.ts";

import { CurrentToolContext } from "./tools/CurrentTool.ts";

import "./Canvas.css";
import { CanvasDragAction } from "./tools/Tool.ts";

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

function makeCanvasPointerEvent(
  event: IPointerEvent,
  camera: Camera2,
  canvas: HTMLElement,
  documentManager: DocumentManager,
): CanvasPointerEvent {
  return {
    camera: camera,
    viewPosition: getMouseViewPosition(event, canvas),
    documentPosition: getMouseDocumentPosition(event, canvas, camera),
    documentManager: documentManager,
    shiftKey: event.shiftKey,
  };
}

export function Canvas({ documentManager }: CanvasProps) {
  const [camera, setCamera] = useState<Camera2>(new Camera2());
  const [pointerState, setPointerState] = useState<PointerState | null>(null);
  const { currentTool } = useContext(CurrentToolContext);

  const ref = useRef<HTMLCanvasElement | null>(null);

  // Note: it's important to use useState and not useRef to store the current
  // dragAction, otherwise in React strict mode, the dragAction could be
  // executed twice causing potential conflicts.
  //
  const [dragAction, setDragAction] = useState<CanvasDragAction | undefined>(
    undefined,
  );

  // Returns whether there is a drag action available for the drag button.
  //
  const onDragStart = useCallback(
    (event: IPointerEvent) => {
      const canvas = ref.current;
      if (!canvas) {
        return false;
      }
      if (!pointerState) {
        return false;
      }
      switch (pointerState.button) {
        case 0: {
          if (!dragAction && currentTool?.onCanvasDrag) {
            const canvasEvent = makeCanvasPointerEvent(
              event,
              camera,
              canvas,
              documentManager,
            );
            let newDragAction = currentTool.onCanvasDrag(canvasEvent);
            if (newDragAction) {
              const started = newDragAction.start();
              if (!started) {
                newDragAction = undefined;
              }
            }
            if (newDragAction) {
              setDragAction(newDragAction);
              return true;
            } else {
              return false;
            }
          }
          return false;
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
    [pointerState, camera, documentManager, currentTool, dragAction],
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
          if (dragAction) {
            dragAction.move(docDelta);
          }
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
    [pointerState, dragAction],
  );

  const onDragEnd = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    (_event: IPointerEvent) => {
      if (!pointerState) {
        return;
      }
      switch (pointerState.button) {
        case 0: {
          if (dragAction) {
            dragAction.end();
            setDragAction(undefined);
          }
          break;
        }
      }
    },
    [pointerState, dragAction],
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
      if (event.button == 2) {
        // right click: reset rotation
        const anchor = getMouseViewPosition(event, canvas);
        const nextCamera = pointerState.cameraOnPress.clone();
        nextCamera.setRotationAround(anchor, 0);
        setCamera(nextCamera);
      } else if (event.button == 0 && currentTool?.onCanvasClick) {
        // left click: tool action
        // Note: for now, we assume that the click action of the tool
        // is for the left click. We may want to generalize this later.
        const canvasEvent = makeCanvasPointerEvent(
          event,
          camera,
          canvas,
          documentManager,
        );
        currentTool.onCanvasClick(canvasEvent);
      }
    },
    [pointerState, documentManager, camera, currentTool],
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
      if (currentTool?.onCanvasHover) {
        const canvasEvent = makeCanvasPointerEvent(
          event,
          camera,
          canvas,
          documentManager,
        );
        currentTool.onCanvasHover(canvasEvent);
      }
    },
    [pointerState, documentManager, camera, currentTool],
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
