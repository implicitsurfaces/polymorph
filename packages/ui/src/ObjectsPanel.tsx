import { Point, SceneManager } from './Scene.ts';
import { NumberInput } from '@ark-ui/react/number-input';

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
            <NumberInput.Root
              value={p.position.x}
              onValueChange={d => {
                sceneManager.scene().points[index].position.x = d.value;
                sceneManager.commitChanges();
              }}
            >
              <NumberInput.Scrubber>
                <NumberInput.Label>X</NumberInput.Label>
              </NumberInput.Scrubber>
              <NumberInput.Input />
            </NumberInput.Root>
            <NumberInput.Root
              value={p.position.y}
              onValueChange={d => {
                sceneManager.scene().points[index].position.y = d.value;
                sceneManager.commitChanges();
              }}
            >
              <NumberInput.Scrubber>
                <NumberInput.Label>Y</NumberInput.Label>
              </NumberInput.Scrubber>
              <NumberInput.Input />
            </NumberInput.Root>
          </div>
        ))}
      </div>
    </div>
  );
}

export default ObjectsPanel;
