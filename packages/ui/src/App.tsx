import { useState, useCallback, useEffect } from 'react';
import { Canvas } from './Canvas.tsx';
import { DocumentManager } from './Document.ts';
import { ObjectsPanel } from './ObjectsPanel.tsx';
import './App.css';

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
    <>
      <ObjectsPanel documentManager={documentManager} />
      <Canvas documentManager={documentManager} />
      <Canvas documentManager={documentManager} />
      <ObjectsPanel documentManager={documentManager} />
    </>
  );
}

export default App;
