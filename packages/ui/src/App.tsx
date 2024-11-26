import { useState, useCallback, useEffect } from 'react';
import { Panel, PanelGroup, PanelResizeHandle, PointerHitAreaMargins } from 'react-resizable-panels';

import { DocumentManager } from './Document.ts';

import { Canvas } from './Canvas.tsx';
import { ObjectsPanel } from './ObjectsPanel.tsx';

import './App.css';
import './Panel.css';

function panelHitMargins(): PointerHitAreaMargins {
  // separator (0-2px) + 2 * margins (3px) = 6-8px total hit area
  return { coarse: 3, fine: 3 };
}

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

  // Application-wide shortcuts.

  const onKeyPress = useCallback(
    (event: KeyboardEvent) => {
      if (event.ctrlKey && (event.key === 'z' || event.key === 'Z')) {
        // Prevent browser doing its own undoing of input fields,
        // interfering with our undo/redo mechanism.
        event.preventDefault();

        if (event.shiftKey) {
          documentManager.redo();
        } else {
          documentManager.undo();
        }
      }
    },
    [documentManager]
  );

  useEffect(() => {
    document.addEventListener('keydown', onKeyPress);
    return () => {
      document.removeEventListener('keydown', onKeyPress);
    };
  }, [onKeyPress]);

  return (
    <PanelGroup className="root-panel-group" direction="vertical">
      <Panel>
        <PanelGroup className="canvas-panel-group" direction="horizontal">
          <Panel defaultSize={50} minSize={10}>
            <Canvas documentManager={documentManager} />
          </Panel>
          <PanelResizeHandle hitAreaMargins={panelHitMargins()} />
          <Panel minSize={10}>
            <Canvas documentManager={documentManager} />
          </Panel>
        </PanelGroup>
      </Panel>
      <PanelResizeHandle hitAreaMargins={panelHitMargins()} />
      <Panel>
        <PanelGroup className="panels-panel-group" direction="horizontal">
          <Panel defaultSize={30} minSize={2}>
            <ObjectsPanel documentManager={documentManager} />
          </Panel>
          <PanelResizeHandle hitAreaMargins={panelHitMargins()} />
          <Panel minSize={2}>
            <div className="panel" />
          </Panel>
        </PanelGroup>
      </Panel>
    </PanelGroup>
  );
}

export default App;
