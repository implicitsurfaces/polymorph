import { useCallback } from "react";
import { useDocumentManager } from "../doc/DocumentManagerContext";
import { useViewContext } from "../view";
import { Canvas, CanvasSettings } from "./Canvas";

import { Panel, PanelGroup, PanelResizeHandle } from "./Panel";

export function CanvasPanels() {
  const documentManager = useDocumentManager();
  const { view, setView } = useViewContext();

  const setLeftCanvasSettings = useCallback(
    (newSettings: CanvasSettings) =>
      setView({ ...view, leftCanvasSettings: newSettings }),
    [view, setView],
  );

  const setRightCanvasSettings = useCallback(
    (newSettings: CanvasSettings) =>
      setView({ ...view, rightCanvasSettings: newSettings }),
    [view, setView],
  );

  if (view.sideBySideCanvas) {
    return (
      <PanelGroup className="canvas-panel-group" direction="horizontal">
        <Panel defaultSize={50} minSize={10}>
          <Canvas
            documentManager={documentManager}
            settings={view.leftCanvasSettings}
            setSettings={setLeftCanvasSettings}
          />
        </Panel>
        <PanelResizeHandle />
        <Panel minSize={10}>
          <Canvas
            documentManager={documentManager}
            settings={view.rightCanvasSettings}
            setSettings={setRightCanvasSettings}
          />
        </Panel>
      </PanelGroup>
    );
  } else {
    return (
      <Canvas
        documentManager={documentManager}
        settings={view.leftCanvasSettings}
        setSettings={setLeftCanvasSettings}
      />
    );
  }
}
