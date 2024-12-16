import { useRef, forwardRef, useState, useCallback, useEffect } from "react";
import { styled } from "goober";
import { observer } from "mobx-react";

import Splitter, { GutterTheme, SplitDirection } from "@devbookhq/splitter";

import { HeaderSelect, Spacer } from "./panes";

import useEditorStore from "../state/useEditorStore";

const Canvas = styled("canvas", forwardRef)`
  position: absolute;
  touch-action: none;
  width: 100%;
  height: 100%;
  ${({ cursor }) => cursor && `cursor: ${cursor};`}
`;

const Image = observer(() => {
  const store = useEditorStore();
  const canvasRef = useRef(null);

  if (store.currentImage && canvasRef?.current) {
    canvasRef.current.width = store.currentDefinition;
    canvasRef.current.height = store.currentDefinition;
    canvasRef.current
      .getContext("2d")
      ?.putImageData(
        new ImageData(
          store.currentImage,
          store.currentDefinition,
          store.currentDefinition,
        ),
        0,
        0,
      );
  }

  return <Canvas ref={canvasRef}></Canvas>;
});

const useCursorType = (hoveredPoint, draggedPoint) => {
  const [shiftPressed, setShiftPressed] = useState(false);
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === "Shift") {
        setShiftPressed(true);
      }
    };

    const handleKeyUp = (e) => {
      if (e.key === "Shift") {
        setShiftPressed(false);
      }
    };

    // Handle cases where key up might occur outside window
    const handleBlur = () => {
      setShiftPressed(false);
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    window.addEventListener("blur", handleBlur);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
      window.removeEventListener("blur", handleBlur);
    };
  }, []);

  if (hoveredPoint && shiftPressed) {
    return 'url("/delete-icon.svg"), auto';
  }

  if (draggedPoint) {
    return "grabbing";
  }

  if (hoveredPoint) {
    return "grab";
  }

  return "cursor";
};

const canvasToImageCoords = (rect, x, y) => {
  const { left, top, width, height } = rect;
  return [(2 * (x - left)) / width - 1, (2 * (top - y)) / height + 1];
};

const imageToCanvasCoords = (x, y) => {
  return [((x + 1) * 1000) / 2, ((1 - y) * 1000) / 2];
};

const PointsLayer = observer(() => {
  const store = useEditorStore();
  const [canvas, setCanvas] = useState(null);

  const [hoveredPoint, setHoveredPoint] = useState(null);
  const [draggedPoint, setDraggingPoint] = useState(null);

  const cursorType = useCursorType(hoveredPoint, draggedPoint);

  if (canvas) {
    const circleCtx = canvas.getContext("2d");

    // Clear and redraw only the points layer
    circleCtx.clearRect(0, 0, canvas.width, canvas.height);

    store.points.forEach((point) => {
      const [x, y] = imageToCanvasCoords(point.x, point.y);
      circleCtx.beginPath();
      circleCtx.arc(x, y, 10, 0, Math.PI * 2);
      circleCtx.fillStyle = point === hoveredPoint ? "white" : "black";
      circleCtx.fill();
      circleCtx.strokeStyle = point === hoveredPoint ? "black" : "white";
      circleCtx.stroke();
    });
  }

  const handleMouseDown = useCallback(
    (e) => {
      const rect = canvas.getBoundingClientRect();
      const point = canvasToImageCoords(rect, e.clientX, e.clientY);

      const closePoint = store.findClosePoint(point);

      if (closePoint) {
        if (e.shiftKey) {
          closePoint.remove();
          setHoveredPoint(null);
        } else {
          setDraggingPoint(closePoint);
          e.target.setPointerCapture(e.pointerId);
        }
        return;
      }

      store.addPoint(point);
    },
    [canvas, store],
  );

  const handleMouseUp = useCallback((e) => {
    e.target.releasePointerCapture(e.pointerId);
    setDraggingPoint(null);
  }, []);

  const handleMouseMove = useCallback(
    (e) => {
      const rect = canvas.getBoundingClientRect();
      const point = canvasToImageCoords(rect, e.clientX, e.clientY);

      if (draggedPoint !== null) {
        draggedPoint.moveTo(point);
      } else {
        // Check for hover
        setHoveredPoint(store.findClosePoint(point));
      }
    },
    [canvas, store, draggedPoint],
  );

  return (
    <Canvas
      ref={setCanvas}
      cursor={cursorType}
      width="1000"
      height="1000"
      onPointerDown={handleMouseDown}
      onPointerMove={handleMouseMove}
      onPointerUp={handleMouseUp}
    ></Canvas>
  );
});

const ValueReadWrapper = styled("div")`
  display: flex;
  flex-direction: column;
  width: 7rem;
  font-size: 0.8em;

  & > :first-child {
    font-weight: bold;
    width: 100%;
    text-overflow: ellipsis;
    white-space: nowrap;
    overflow: hidden;
  }
`;

const fmt = (v) => {
  return Intl.NumberFormat("en", {
    maximumSignificantDigits: 3,
  }).format(Math.round(v * 1e6) / 1e6);
};

function ValueRead({ name, value }) {
  if (value === null) return null;
  const valueTxt = Array.isArray(value)
    ? `(${value.map(fmt).join(", ")})`
    : fmt(value);
  return (
    <ValueReadWrapper>
      <span>{name}</span>
      <span>{valueTxt}</span>
    </ValueReadWrapper>
  );
}

const ValueReadsWrapper = styled("div")`
  position: relative;
  overflow-y: auto;
  width: 100%;
  display: flex;
  flex-wrap: wrap;
  gap: 0.4em;
  padding: 0.9em;
`;

const SplitterContainer = styled("div")`
  overflow: hidden;
  width: 100%;
  height: 100%;
`;

const CanvasesContainer = styled("div")`
  position: relative;
  display: flex;
  max-width: 100%;
  aspect-ratio: 1;

  margin: 0 auto auto 0;
`;

export const VisualizerPane = observer(() => {
  const store = useEditorStore();
  return (
    <Splitter
      direction={SplitDirection.Vertical}
      gutterTheme={GutterTheme.Dark}
      gutterClassName="custom-gutter-theme"
      initialSizes={store.valueReads.length ? [75, 25] : [100]}
    >
      <SplitterContainer>
        <CanvasesContainer>
          <Image />
          <PointsLayer />
        </CanvasesContainer>
      </SplitterContainer>
      {!!store.valueReads.length && (
        <ValueReadsWrapper>
          {store.valueReads.map((value, index) => (
            <ValueRead key={index} {...value} />
          ))}
        </ValueReadsWrapper>
      )}
    </Splitter>
  );
});

const DEFINITONS = [
  { name: "Very Low", value: 250 },
  { name: "Low", value: 750 },
  { name: "Medium", value: 1500 },
  { name: "High", value: 3000 },
];

export const VisualizerButtons = observer(() => {
  const store = useEditorStore();

  return (
    <>
      <>
        <HeaderSelect
          value={store.definition}
          onChange={(e) => store.changeDefinition(parseInt(e.target.value))}
        >
          {DEFINITONS.map(({ name, value }) => (
            <option value={value} key={value}>
              {name}
            </option>
          ))}
        </HeaderSelect>
        <Spacer />
      </>
    </>
  );
});
