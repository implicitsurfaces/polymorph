import { useState, useCallback, useEffect } from 'react';
import { Canvas } from './Canvas.tsx';
import { DocumentManager } from './Document.ts';
import { ObjectsPanel } from './ObjectsPanel.tsx';
import './App.css';

let _uniqueVersion = 0;
function generateUniqueVersion() {
  _uniqueVersion += 1;
  return _uniqueVersion;
}

function App() {
  const [version, setVersion] = useState(generateUniqueVersion());

  // Update the version whenever the working document changes.
  //
  // This ensures we re-render the App despite having reference-equality of
  // documentManager, since we allow direct mutations for performance
  // reasons, especially during mouse drags.
  //
  // Note that this callback has no dependencies, since `setVersion` has
  // stable identity for the lifetime of the component, which is a guarantee
  // provided by the `useState()` hook.
  //
  const onDocumentChange = useCallback(() => {
    setVersion(generateUniqueVersion());
  }, []);

  // Create the DocumentManager.
  //
  // It has stable identity but can be mutated, in which case
  // `onDocumentChange` is called.
  //
  const [documentManager] = useState(new DocumentManager(onDocumentChange));

  // Application-wide shortcuts.

  const onKeyPress = useCallback(
    (event: KeyboardEvent) => {
      // Prevent browser doing its own undoing of input fields,
      // interfering with our undo/redo mechanism.
      event.preventDefault();

      if (event.ctrlKey && (event.key === 'z' || event.key === 'Z')) {
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
      <Canvas documentManager={documentManager} version={version} />
      <Canvas documentManager={documentManager} version={version} />
      <ObjectsPanel documentManager={documentManager} />
    </>
  );
}

export default App;
