import { Point, SceneManager } from './Scene.ts';

import './Panel.css';
import './ObjectsPanel.css';

interface ObjectsPanelProps {
  sceneManager: SceneManager;
}

export function ObjectsPanel({ sceneManager }: ObjectsPanelProps) {
  return (
    <div className="panel">
      <h2 className="panel-title">Objects</h2>
      <div className="panel-body">
        {sceneManager.scene().points.map((p: Point) => (
          <div className="object-row-info" key={p.name}>
            <p>{p.name}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default ObjectsPanel;
