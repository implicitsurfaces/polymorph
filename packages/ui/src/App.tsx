import { useState } from 'react';
import { Canvas } from './Canvas.tsx';
import { Scene } from './Scene.ts';
import './App.css';

function App() {
  const [scene, setScene] = useState(new Scene());

  return (
    <>
      <Canvas scene={scene} setScene={setScene} />
      <Canvas scene={scene} setScene={setScene} />
    </>
  );
}

export default App;
