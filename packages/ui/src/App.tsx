import { useState, useCallback, useEffect } from 'react';
import { Canvas } from './Canvas.tsx';
import { DocumentManager } from './Document.ts';
import { ObjectsPanel } from './ObjectsPanel.tsx';
import './App.css';

function App() {
  // Create and setup the DocumentManager.
  //
  // Note that the onChange callback stored on the DocumentManager recursively
  // depends on documentManager itself (and on setDocumentManager), which is why we
  // set it by directly mutating documentManager after the useState() call.
  //
  // This is technically a violation of React immutability assumption, but in
  // practice it isn't a problem in this case since the onChange callback is
  // (normally) never called at React render time, but only during mouse/key
  // event processing, which happens after React is done rendering
  // (generating the virtual DOM and updating the actual DOM).
  //
  // It would be more "pure" not to store the onChange callback within the
  // documentManager state, but instead pass it to child components as a
  // separate prop. However, doing so would lead to an API that isn't as
  // convenient as passing a single DocumentManager that knows itself how to
  // call onChange whenever relevant.
  //
  const [documentManager, setDocumentManager] = useState(new DocumentManager());
  documentManager.onChange(() => {
    setDocumentManager(documentManager.shallowClone());
  });

  // Application-wide shortcuts.

  const onKeyPress = useCallback(
    (event: KeyboardEvent) => {
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
      <Canvas documentManager={documentManager} />
      <Canvas documentManager={documentManager} />
      <ObjectsPanel documentManager={documentManager} />
    </>
  );
}

export default App;
