import { useState, useCallback, useEffect, useRef, useMemo } from "react";
import {
  Panel,
  PanelGroup,
  PanelResizeHandle,
  PointerHitAreaMargins,
} from "react-resizable-panels";

import { DocumentManager } from "./doc/DocumentManager";

import { TriggerAction } from "./actions/Action";
import { Tool } from "./tools/Tool";
import { Toolbar } from "./tools/Toolbar";
import { CurrentTool, CurrentToolContext } from "./tools/CurrentTool";
import { allActions } from "./allActions";

import { Canvas, CanvasSettings } from "./components/Canvas";
import { LayersPanel } from "./components/LayersPanel";
import { SkeletonPanel, MeasuresPanel } from "./components/NodeListPanel";
import { PropertiesPanel } from "./components/PropertiesPanel";

import "./App.css";
import "./components/Panel.css";

function panelHitMargins(): PointerHitAreaMargins {
  // separator (0-2px) + 2 * margins (3px) = 6-8px total hit area
  return { coarse: 3, fine: 3 };
}

type ClientSize = {
  width: number;
  height: number;
};

function App() {
  // Create the DocumentManager.
  //
  // It has stable identity and stores the document history as well as a
  // mutable working copy.
  //
  // Each time the working copy is changed, `documentManager.version()` is
  // incremented and the callback given to `documentManager.onChange()` is
  // called.
  //
  const [documentManager] = useState(() => new DocumentManager());

  // Store a local copy of the version and update it on document change.
  //
  // This ensures we re-render the App despite having reference-equality of
  // documentManager.
  //
  const [, setVersion] = useState(documentManager.version());

  const onDocumentChange = useCallback(() => {
    setVersion(documentManager.version());
  }, [documentManager]);

  documentManager.onChange(onDocumentChange);

  // Actions and Tools
  const [actions] = useState(allActions());
  const [currentTool, setCurrentTool] = useState<CurrentTool>(() => {
    for (const action of actions) {
      if (action instanceof Tool) {
        return action;
      }
    }
  });

  // Application-wide shortcuts
  const onKeyPress = useCallback(
    (event: KeyboardEvent) => {
      for (const action of actions) {
        if (action.shortcut && action.shortcut.matches(event)) {
          if (action instanceof TriggerAction) {
            // Prevent browser doing its own thing (e.g., its own implementation
            // of undo/redo, etc.)
            event.preventDefault();
            action.onTrigger(documentManager);
            return;
          } else if (action instanceof Tool) {
            setCurrentTool(action);
            return;
          }
        }
      }
    },
    [documentManager, actions],
  );

  useEffect(() => {
    document.addEventListener("keydown", onKeyPress);
    return () => {
      document.removeEventListener("keydown", onKeyPress);
    };
  }, [onKeyPress]);

  // react-resizable-panels does not support setting min/default sizes
  // as pixels instead of percentages, so we use this as a workaround.
  //
  // See:
  // - https://github.com/bvaughn/react-resizable-panels/issues/46
  // - https://github.com/bvaughn/react-resizable-panels/pull/176
  // - https://stackoverflow.com/questions/69819938
  //
  const [clientSize, setClientSize] = useState<ClientSize | null>(null);

  const ref = useRef<HTMLDivElement>(null);

  const elementObserver = useMemo(() => {
    return new ResizeObserver(() => {
      const element = ref.current;
      if (!element) {
        return null;
      }
      setClientSize({
        height: element.clientHeight,
        width: element.clientWidth,
      });
    });
  }, []);

  useEffect(() => {
    const element = ref.current;
    if (!element || !elementObserver) {
      return () => {};
    }
    elementObserver.observe(element);
    return () => {
      elementObserver.unobserve(element);
    };
  }, [elementObserver]);

  // Canvas settings
  const [leftCanvasSettings] = useState(() => new CanvasSettings());
  const [rightCanvasSettings] = useState(
    () => new CanvasSettings({ sdfTest: true }),
  );

  // Do not create any Panel components until we know the size of the app via
  // the ResizeObserver, so that we can provide an appropriate defaultSize to
  // the panels, which must be done on first creation of the Panel.
  //
  if (!clientSize) {
    return <div className="app" ref={ref} />;
  }

  // Compute the desired panel min width in pixels
  //
  // TODO: get separatorWidth and numPanels programatically
  //
  const separatorWidth = 1;
  const numPanels = 1; // not including the leftover panel
  const numSeparators = numPanels;
  const totalPanelWidth = clientSize.width - numSeparators * separatorWidth;

  let panelMinWidth = 50;
  if (numPanels > 0 && totalPanelWidth < panelMinWidth * numPanels) {
    panelMinWidth = totalPanelWidth / numPanels;
  }

  let panelDefaultWidth = 250;
  if (numPanels > 0 && totalPanelWidth < panelDefaultWidth * numPanels) {
    panelDefaultWidth = totalPanelWidth / numPanels;
  }

  // Convert from pixels to percentage.
  //
  // Note: react-resizable-panels uses precision=3 to convert the percentage
  // to CSS, e.g.: size=51.426 becomes style="flex: 51.4 1 0px;".
  //
  // This means that the panel will not end up having exactly the prescribed
  // pixel size. Usually it's fine since the browser will round anyway, but
  // it'd be nice if the chosen precision was exposed. It's defined here:
  //
  // react-resizable-panels/src/utils/computePanelFlexBoxStyle.ts
  //
  let panelMinSize = 1;
  let panelDefaultSize = 1;
  if (totalPanelWidth > 1) {
    panelMinSize = (panelMinWidth / totalPanelWidth) * 100;
    panelDefaultSize = (panelDefaultWidth / totalPanelWidth) * 100;
  }

  return (
    <div className="app" ref={ref}>
      <CurrentToolContext.Provider
        value={{
          currentTool,
          setCurrentTool,
        }}
      >
        <Toolbar actions={actions} documentManager={documentManager} />
        <PanelGroup className="root-panel-group" direction="vertical">
          <Panel>
            <PanelGroup className="canvas-panel-group" direction="horizontal">
              <Panel defaultSize={50} minSize={10}>
                <Canvas
                  documentManager={documentManager}
                  settings={leftCanvasSettings}
                />
              </Panel>
              <PanelResizeHandle hitAreaMargins={panelHitMargins()} />
              <Panel minSize={10}>
                <Canvas
                  documentManager={documentManager}
                  settings={rightCanvasSettings}
                />
              </Panel>
            </PanelGroup>
          </Panel>
          <PanelResizeHandle hitAreaMargins={panelHitMargins()} />
          <Panel>
            <PanelGroup className="panels-panel-group" direction="horizontal">
              <Panel defaultSize={panelDefaultSize} minSize={panelMinSize}>
                <LayersPanel documentManager={documentManager} />
              </Panel>
              <PanelResizeHandle hitAreaMargins={panelHitMargins()} />
              <Panel defaultSize={panelDefaultSize} minSize={panelMinSize}>
                <SkeletonPanel documentManager={documentManager} />
              </Panel>
              <PanelResizeHandle hitAreaMargins={panelHitMargins()} />
              <Panel defaultSize={panelDefaultSize} minSize={panelMinSize}>
                <MeasuresPanel documentManager={documentManager} />
              </Panel>
              <PanelResizeHandle hitAreaMargins={panelHitMargins()} />
              <Panel defaultSize={panelDefaultSize} minSize={panelMinSize}>
                <PropertiesPanel documentManager={documentManager} />
              </Panel>
              <PanelResizeHandle hitAreaMargins={panelHitMargins()} />
              <Panel minSize={0}>
                <div className="panel leftover" />
              </Panel>
            </PanelGroup>
          </Panel>
        </PanelGroup>
      </CurrentToolContext.Provider>
    </div>
  );
}

export default App;
