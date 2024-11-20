import { Point, DocumentManager } from './Document.ts';
import { NumberInput } from './NumberInput.tsx';

import './Panel.css';
import './ObjectsPanel.css';

interface ObjectsPanelProps {
  documentManager: DocumentManager;
}

export function ObjectsPanel({ documentManager }: ObjectsPanelProps) {
  return (
    <div className="panel">
      <h2 className="panel-title">Objects</h2>
      <div className="panel-body">
        {documentManager.document().points.map((p: Point, index: number) => (
          <div className="object-row-info" key={p.name}>
            <p className="object-name">{p.name}</p>
            <NumberInput
              idBase={`number-input::x${index}`}
              label="X"
              value={p.position.x}
              onChange={value => {
                documentManager.document().points[index].position.x = value;
                documentManager.commitChanges();
              }}
            />
            <NumberInput
              idBase={`number-input::y${index}`}
              label="Y"
              value={p.position.y}
              onChange={value => {
                documentManager.document().points[index].position.y = value;
                documentManager.commitChanges();
              }}
            />
          </div>
        ))}
      </div>
    </div>
  );
}

export default ObjectsPanel;
