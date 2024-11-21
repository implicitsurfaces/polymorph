import React, { useEffect, useRef } from "react";
import { styled } from "goober";
import { observer } from "mobx-react";
import Splitter, { GutterTheme } from "@devbookhq/splitter";

import useEditorStore, { EditorContextProvider } from "./state/useEditorStore";

import { Pane } from "./components/panes";
import { EditorPane } from "./components/EditorPane";

const Canvas = styled("canvas", React.forwardRef)`
  width: 100%;
  aspect-ratio: 1;
  margin: auto;
`;

const VisualizerPane = observer(() => {
  const store = useEditorStore();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  if (canvasRef?.current && store.currentImage) {
    canvasRef.current
      .getContext("2d")
      ?.putImageData(
        new ImageData(store.currentImage, store.definition, store.definition),
        0,
        0,
      );
  }

  return (
    <Canvas
      width={store.definition}
      height={store.definition}
      ref={canvasRef}
    ></Canvas>
  );
});

export const WorkbenchStructure = observer(function WorkbenchStructure() {
  const store = useEditorStore();
  useEffect(() => {
    store.initCode();
  }, [store]);

  return (
    <>
      <Splitter
        gutterTheme={GutterTheme.Dark}
        gutterClassName="custom-gutter-theme"
      >
        <Pane aboveOthers>
          <EditorPane />
        </Pane>
        <Pane>
          <VisualizerPane />
        </Pane>
      </Splitter>
    </>
  );
});

const WorkbenchWrapper = styled("div")`
  height: 100vh;
  width: 100vw;
  max-height: 100vh;
  max-width: 100vw;
  display: flex;
  flex-direction: column;
  position: relative;
  overflow-y: hidden;

  & .custom-gutter-theme {
    background-color: var(--color-primary-light);
  }
`;

function Workbench() {
  return (
    <WorkbenchWrapper>
      <EditorContextProvider>
        <WorkbenchStructure />
      </EditorContextProvider>
    </WorkbenchWrapper>
  );
}

export default Workbench;
