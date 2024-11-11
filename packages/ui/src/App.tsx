import { useState, useCallback, useEffect } from 'react';
import { Canvas } from './Canvas.tsx';
import { SceneManager } from './Scene.ts';
import './App.css';

function App() {
  // Create and setup the SceneManager.
  //
  // Note that the onChange callback stored on the SceneManager recursively
  // depends on sceneManager itself (and on setSceneManager), which is why we
  // set it by directly mutating sceneManager after the useState() call.
  //
  // This is technically a violation of React immutability assumption, but in
  // practice it isn't a problem in this case since the onChange callback is
  // (normally) never called at React render time, but only during mouse/key
  // event processing, which happens after React is done rendering
  // (generating the virtual DOM and updating the actual DOM).
  //
  // It would be more "pure" not to store the onChange callback within the
  // sceneManager state, but instead pass it to child components as a
  // separate prop. However, doing so would lead to an API that isn't as
  // convenient as passing a single SceneManager that knows itself how to
  // call onChange whenever relevant.
  //
  const [sceneManager, setSceneManager] = useState(new SceneManager());
  sceneManager.onChange(() => {
    setSceneManager(sceneManager.shallowClone());
  });

  // Application-wide shortcuts.

  const onKeyPress = useCallback(
    event => {
      if (event.ctrlKey && (event.key === 'z' || event.key === 'Z')) {
        if (event.shiftKey) {
          sceneManager.redo();
        } else {
          sceneManager.undo();
        }
      }
    },
    [sceneManager]
  );

  useEffect(() => {
    document.addEventListener('keydown', onKeyPress);
    return () => {
      document.removeEventListener('keydown', onKeyPress);
    };
  }, [onKeyPress]);

  return (
    <>
      <Canvas sceneManager={sceneManager} />
      <Canvas sceneManager={sceneManager} />
    </>
  );
}

export default App;
