import { Point, SceneManager } from './Scene.ts';
import { NumberInput } from './NumberInput.tsx';

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
        {sceneManager.scene().points.map((p: Point, index: number) => (
          <div className="object-row-info" key={p.name}>
            <p className="object-name">{p.name}</p>
            <NumberInput
              idBase={`number-input::x${index}`}
              label="X"
              value={p.position.x}
              onChange={value => {
                sceneManager.scene().points[index].position.x = value;
                sceneManager.commitChanges();
              }}
            />
            <NumberInput
              idBase={`number-input::y${index}`}
              label="Y"
              value={p.position.y}
              onChange={value => {
                sceneManager.scene().points[index].position.y = value;
                sceneManager.commitChanges();
              }}
            />
          </div>
        ))}
      </div>
    </div>
  );
}

export default ObjectsPanel;
